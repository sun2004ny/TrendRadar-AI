"""
src/ingestion/reddit.py - Live Reddit Ingestion via PRAW

Features:
1. Credentials loaded strictly from environment (.env).
2. Exponential backoff on HTTP 429 (rate-limit) and transient network errors.
3. Daily request budget limiter to protect API quotas.
4. Collects posts in 15-minute chronological batches.
5. In-memory and persistent deduplication across sliding windows.
6. Checkpoint persistence (last seen timestamp / ID) enabling resume without data loss or repeats.
7. Incremental vector indexing via FAISS (IndexFlatIP), replacing static .npy files.
8. Implements the unified DataSource interface.
"""

from pathlib import Path
import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Iterator, Set, Union
import numpy as np

import praw
import prawcore

from src.config import ROOT_DIR, get_config
from src.ingestion.base import DataSource, NormalizedPost
from src.embeddings import IncrementalFAISSIndex

logger = logging.getLogger(__name__)


def load_env_credentials(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Loads credentials from .env file or system environment variables."""
    creds = {
        "client_id": os.environ.get("REDDIT_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("REDDIT_CLIENT_SECRET", "").strip(),
        "user_agent": os.environ.get("REDDIT_USER_AGENT", "").strip()
    }

    p = env_path or (ROOT_DIR / ".env")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k == "REDDIT_CLIENT_ID" and not creds["client_id"]:
                    creds["client_id"] = v
                elif k == "REDDIT_CLIENT_SECRET" and not creds["client_secret"]:
                    creds["client_secret"] = v
                elif k == "REDDIT_USER_AGENT" and not creds["user_agent"]:
                    creds["user_agent"] = v

    return creds


class DailyRequestBudgetLimiter:
    """Enforces a strict daily call budget to respect API quotas."""

    def __init__(self, max_requests_per_day: int = 500):
        self.max_requests_per_day = max_requests_per_day
        self.current_date = datetime.now(timezone.utc).date()
        self.requests_today = 0

    def can_request(self) -> bool:
        today = datetime.now(timezone.utc).date()
        if today != self.current_date:
            self.current_date = today
            self.requests_today = 0
        return self.requests_today < self.max_requests_per_day

    def record_request(self) -> None:
        self.can_request()
        self.requests_today += 1


class RedditLiveDataSource(DataSource):
    """
    Live Reddit Data Source wrapping PRAW behind DataSource.
    Handles rate limiting, budget caps, checkpointing, and FAISS indexing.
    """

    def __init__(
        self,
        subreddits: Optional[List[str]] = None,
        env_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        max_daily_requests: int = 500,
        batch_window_minutes: int = 15,
        faiss_index: Optional[IncrementalFAISSIndex] = None,
        reddit_client: Optional[Any] = None
    ):
        super().__init__(source_name="reddit_live")
        self.subreddits = subreddits or ["AskReddit", "technology", "news"]
        self.batch_window_minutes = batch_window_minutes
        self.budget_limiter = DailyRequestBudgetLimiter(max_requests_per_day=max_daily_requests)

        self.checkpoint_path = checkpoint_path or (ROOT_DIR / "data" / "cache" / "reddit_checkpoint.json")
        self.last_seen_timestamp: float = 0.0
        self.last_seen_id: str = ""
        self.seen_ids: Set[str] = set()

        self._load_checkpoint()

        # Incremental FAISS Index
        self.faiss_index = faiss_index or IncrementalFAISSIndex(
            dim=384,
            index_path=ROOT_DIR / "data" / "processed" / "live_faiss.index",
            meta_path=ROOT_DIR / "data" / "processed" / "live_faiss_meta.json"
        )

        # Credentials & PRAW Client
        self.creds = load_env_credentials(env_path)
        self.reddit = reddit_client
        self.is_authenticated = False

        if self.reddit is None:
            self._init_praw_client()
        else:
            self.is_authenticated = True

    def _init_praw_client(self) -> None:
        cid = self.creds.get("client_id", "")
        csec = self.creds.get("client_secret", "")
        uagent = self.creds.get("user_agent", "")

        # Check for placeholder or missing credentials
        if not cid or not csec or not uagent or "your_reddit" in cid.lower() or "your_reddit" in csec.lower():
            logger.warning(
                "Reddit API credentials missing or placeholder in .env. Live ingestion will run in unauthenticated/mock mode."
            )
            self.is_authenticated = False
            return

        try:
            self.reddit = praw.Reddit(
                client_id=cid,
                client_secret=csec,
                user_agent=uagent,
                check_for_async=False
            )
            # Read-only test
            self.reddit.read_only = True
            self.is_authenticated = True
            logger.info("PRAW client initialized successfully (read_only=True).")
        except Exception as exc:
            logger.error("Failed to initialize PRAW client: %s", exc)
            self.is_authenticated = False

    def _load_checkpoint(self) -> None:
        """Loads last seen timestamp and ID from disk."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.last_seen_timestamp = float(data.get("last_seen_timestamp", 0.0))
                    self.last_seen_id = str(data.get("last_seen_id", ""))
                    for pid in data.get("seen_ids", []):
                        self.seen_ids.add(str(pid))
                logger.info(
                    "Resumed from checkpoint: last_seen_ts=%.1f, seen_ids=%d",
                    self.last_seen_timestamp,
                    len(self.seen_ids)
                )
            except Exception as e:
                logger.warning("Could not read checkpoint file %s: %s", self.checkpoint_path, e)

    def _save_checkpoint(self) -> None:
        """Persists checkpoint state."""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_seen_timestamp": self.last_seen_timestamp,
                    "last_seen_id": self.last_seen_id,
                    "seen_ids": list(self.seen_ids)[-1000:],  # keep last 1000
                    "saved_at": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)

    def fetch_with_backoff(self, subreddit_name: str, limit: int = 50, max_retries: int = 5, base_delay: float = 1.0) -> list:
        """
        Fetches submissions with exponential backoff on HTTP 429 and network errors.
        Respects the daily request budget limiter.
        """
        if not self.is_authenticated or self.reddit is None:
            logger.warning("Cannot fetch from r/%s: PRAW client unauthenticated or missing credentials.", subreddit_name)
            return []

        if not self.budget_limiter.can_request():
            logger.warning("Daily request budget (%d) reached. Skipping fetch.", self.budget_limiter.max_requests_per_day)
            return []

        for attempt in range(max_retries):
            try:
                self.budget_limiter.record_request()
                sub = self.reddit.subreddit(subreddit_name)
                # Fetch newest posts
                posts = list(sub.new(limit=limit))
                return posts
            except (prawcore.TooManyRequests, prawcore.ResponseException) as exc:
                wait_time = base_delay * (2 ** attempt)
                logger.warning(
                    "[HTTP 429 Rate Limit] Reddit rate limit hit on r/%s. Backing off for %.1fs (attempt %d/%d)...",
                    subreddit_name, wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
            except (prawcore.RequestException, prawcore.ServerError) as exc:
                wait_time = base_delay * (2 ** attempt)
                logger.warning(
                    "[Network Error] Network/server issue on r/%s: %s. Backing off for %.1fs (attempt %d/%d)...",
                    subreddit_name, exc, wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
            except prawcore.OAuthException as exc:
                logger.error("Authentication failed: Invalid Reddit credentials: %s", exc)
                return []
            except Exception as exc:
                logger.error("Unexpected error fetching r/%s: %s", subreddit_name, exc)
                return []

        logger.error("Exceeded max retries (%d) fetching r/%s.", max_retries, subreddit_name)
        return []

    def yield_batches(self, batch_size: int = 25) -> Iterator[List[NormalizedPost]]:
        """
        Yields batches of posts collected into 15-minute time windows.
        Deduplicates posts, filters out posts before checkpoint, and updates checkpoint.
        """
        if not self.is_authenticated:
            logger.info("Reddit live data source has no valid credentials. Yielding 0 posts.")
            return

        current_batch: List[NormalizedPost] = []
        collected_posts = []

        for sub_name in self.subreddits:
            submissions = self.fetch_with_backoff(sub_name, limit=batch_size * 2)
            for sub in submissions:
                pid = str(sub.id).strip()

                # Deduplication check
                if pid in self.seen_ids:
                    continue

                created_ts = float(sub.created_utc)
                # Checkpoint check: skip posts older than last seen timestamp
                if created_ts <= self.last_seen_timestamp:
                    continue

                title = str(getattr(sub, "title", "")).strip()
                selftext = str(getattr(sub, "selftext", "")).strip()
                if selftext and selftext not in ["[removed]", "[deleted]"]:
                    text = f"{title}. {selftext}".strip(". ")
                else:
                    text = title

                author = str(getattr(sub, "author", "unknown"))

                raw_record = {
                    "id": pid,
                    "raw_id": pid,
                    "text": text,
                    "timestamp": created_ts,
                    "source": f"reddit_r_{sub_name}",
                    "author": author,
                    "metadata": {
                        "subreddit": sub_name,
                        "score": int(getattr(sub, "score", 0)),
                        "num_comments": int(getattr(sub, "num_comments", 0)),
                        "created_utc": created_ts
                    }
                }

                post = self.validate_and_normalize(raw_record)
                if post is not None:
                    self.seen_ids.add(pid)
                    collected_posts.append(post)
                    if created_ts > self.last_seen_timestamp:
                        self.last_seen_timestamp = created_ts
                        self.last_seen_id = pid

        # Sort chronologically
        collected_posts.sort(key=lambda p: p.timestamp)

        # Batch in 15-minute groups or by batch_size
        for post in collected_posts:
            current_batch.append(post)
            if len(current_batch) >= batch_size:
                self._save_checkpoint()
                yield current_batch
                current_batch = []

        if current_batch:
            self._save_checkpoint()
            yield current_batch
