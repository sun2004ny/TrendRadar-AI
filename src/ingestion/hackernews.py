"""
src/ingestion/hackernews.py - Hacker News Live & Mock Ingestion Data Source

Ingests stories from the official, free Hacker News Firebase REST API (no authentication key required).
Provides:
- NormalizedPost output
- Rate limiting with exponential backoff
- Checkpointing and deduplication
- Robust error handling
"""

from pathlib import Path
import sys
import time
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Set, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.base import DataSource, NormalizedPost

logger = logging.getLogger(__name__)

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL_TEMPLATE = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"


class HackerNewsDataSource(DataSource):
    """
    Ingests stories from Hacker News via Firebase API.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        max_stories_per_fetch: int = 50,
        max_stories: Optional[int] = None,
        requests_per_second: float = 5.0,
        delay_between_requests: Optional[float] = None,
        max_retries: int = 3,
        base_backoff: float = 1.0
    ):
        super().__init__(source_name="hackernews")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else ROOT_DIR / "data" / "cache" / "hn_checkpoint.json"
        self.max_stories_per_fetch = max_stories if max_stories is not None else max_stories_per_fetch
        if delay_between_requests is not None:
            self.min_interval = delay_between_requests
        else:
            self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.2
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._last_req_time: float = 0.0

        self.seen_ids: Set[str] = set()
        self._load_checkpoint()


    def _load_checkpoint(self) -> None:
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.seen_ids = set(data.get("seen_ids", []))
            except Exception as e:
                logger.warning("Could not read HN checkpoint: %s", e)

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"seen_ids": sorted(list(self.seen_ids))}, f, indent=2)
            except Exception as e:
                logger.error("Could not save HN checkpoint: %s", e)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_req_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_req_time = time.time()

    def _get_json(self, url: str) -> Optional[Any]:
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "TrendRadar-AI/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 429 or 500 <= e.code < 600:
                    wait_t = self.base_backoff * (2 ** attempt)
                    time.sleep(wait_t)
                else:
                    logger.warning("HTTP error %s on %s", e.code, url)
                    return None
            except Exception as e:
                wait_t = self.base_backoff * (2 ** attempt)
                time.sleep(wait_t)
        return None

    def fetch_top_stories(self) -> List[NormalizedPost]:
        """Fetches top story IDs and yields new, deduplicated NormalizedPost objects."""
        story_ids = self._get_json(HN_TOP_STORIES_URL)
        if not story_ids or not isinstance(story_ids, list):
            return []

        posts: List[NormalizedPost] = []
        fetched_count = 0

        for sid in story_ids:
            if fetched_count >= self.max_stories_per_fetch:
                break

            sid_str = str(sid)
            if sid_str in self.seen_ids:
                continue

            item = self._get_json(HN_ITEM_URL_TEMPLATE.format(item_id=sid))
            if not item or item.get("type") != "story":
                continue

            title = item.get("title", "").strip()
            text = item.get("text", "") or ""
            full_text = f"{title} {text}".strip()

            if not full_text:
                self.invalid_posts_count += 1
                continue

            created_time = item.get("time", int(time.time()))
            meta = {
                "score": item.get("score", 0),
                "url": item.get("url", ""),
                "descendants": item.get("descendants", 0),
                "hn_id": sid
            }

            try:
                post = NormalizedPost(
                    id=f"hn_{sid_str}",
                    raw_id=sid_str,
                    text=full_text,
                    timestamp=created_time,
                    source="hackernews",
                    metadata=meta
                )
                posts.append(post)
                self.seen_ids.add(sid_str)
                self.valid_posts_count += 1
                fetched_count += 1
            except Exception as e:
                self.invalid_posts_count += 1
                logger.debug("Rejected invalid HN post %s: %s", sid_str, e)

        self._save_checkpoint()
        return posts

    def fetch_latest(self) -> List[NormalizedPost]:
        """Alias for fetch_top_stories."""
        return self.fetch_top_stories()

    def yield_batches(self, batch_size: int = 50) -> Iterator[List[NormalizedPost]]:
        """Yields batches of top stories."""
        posts = self.fetch_top_stories()
        for i in range(0, len(posts), batch_size):
            yield posts[i:i + batch_size]

    def stream_batches(self, batch_size: int = 50) -> Iterator[List[NormalizedPost]]:
        """Alias for yield_batches."""
        yield from self.yield_batches(batch_size=batch_size)

