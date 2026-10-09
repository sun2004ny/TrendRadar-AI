"""
src/ingestion/rss.py - News RSS/Atom Feed Ingestion Data Source

Ingests news articles and headlines from standard RSS/Atom feeds.
Provides:
- NormalizedPost output
- Rate limiting with exponential backoff
- Checkpointing and deduplication
- Robust XML parsing and timezone normalization
"""

from pathlib import Path
import sys
import time
import json
import logging
import hashlib
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Set, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.base import DataSource, NormalizedPost

logger = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
]


class RSSDataSource(DataSource):
    """
    Ingests live headlines and articles from public RSS/Atom feeds.
    """

    def __init__(
        self,
        feed_urls: Optional[List[str]] = None,
        checkpoint_path: Optional[Path] = None,
        max_items_per_feed: int = 50,
        max_retries: int = 3,
        base_backoff: float = 1.0
    ):
        super().__init__(source_name="rss")
        self.feed_urls = feed_urls or DEFAULT_FEEDS
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else ROOT_DIR / "data" / "cache" / "rss_checkpoint.json"
        self.max_items_per_feed = max_items_per_feed
        self.max_retries = max_retries
        self.base_backoff = base_backoff

        self.seen_ids: Set[str] = set()
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.seen_ids = set(data.get("seen_ids", []))
            except Exception as e:
                logger.warning("Could not read RSS checkpoint: %s", e)

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"seen_ids": sorted(list(self.seen_ids))}, f, indent=2)
            except Exception as e:
                logger.error("Could not save RSS checkpoint: %s", e)

    def _fetch_xml(self, url: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "TrendRadar-AI/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                if e.code == 429 or 500 <= e.code < 600:
                    time.sleep(self.base_backoff * (2 ** attempt))
                else:
                    logger.warning("HTTP error %s on %s", e.code, url)
                    return None
            except Exception as e:
                time.sleep(self.base_backoff * (2 ** attempt))
        return None

    def fetch_feeds(self) -> List[NormalizedPost]:
        """Fetches all configured feeds and returns new, deduplicated NormalizedPost objects."""
        all_posts: List[NormalizedPost] = []

        for url in self.feed_urls:
            xml_text = self._fetch_xml(url)
            if not xml_text:
                continue

            try:
                root = ET.fromstring(xml_text)
            except Exception as e:
                logger.warning("Failed to parse XML from %s: %s", url, e)
                continue

            # Check RSS <item> vs Atom <entry>
            items = root.findall(".//item")
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

            count = 0
            for item in items:
                if count >= self.max_items_per_feed:
                    break

                # Extract title, description, link, guid/id, pubDate
                title = item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or ""
                desc = item.findtext("description") or item.findtext("{http://www.w3.org/2005/Atom}summary") or ""
                link = item.findtext("link") or item.findtext("{http://www.w3.org/2005/Atom}link") or ""
                guid = item.findtext("guid") or item.findtext("{http://www.w3.org/2005/Atom}id") or link or title

                guid_hash = hashlib.sha256(guid.encode("utf-8")).hexdigest()[:16]
                if guid_hash in self.seen_ids:
                    continue

                full_text = f"{title.strip()} {desc.strip()}".strip()
                if not full_text:
                    self.invalid_posts_count += 1
                    continue

                # Parse date
                pub_date_str = item.findtext("pubDate") or item.findtext("{http://www.w3.org/2005/Atom}updated") or ""
                dt = datetime.now(timezone.utc)
                if pub_date_str:
                    try:
                        dt = parsedate_to_datetime(pub_date_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                    except Exception:
                        try:
                            dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        except Exception:
                            dt = datetime.now(timezone.utc)

                try:
                    p = NormalizedPost(
                        id=f"rss_{guid_hash}",
                        raw_id=guid_hash,
                        text=full_text,
                        timestamp=dt,
                        source="rss",
                        metadata={"feed_url": url, "link": link}
                    )
                    all_posts.append(p)
                    self.seen_ids.add(guid_hash)
                    self.valid_posts_count += 1
                    count += 1
                except Exception as e:
                    self.invalid_posts_count += 1
                    logger.debug("Rejected invalid RSS post %s: %s", guid_hash, e)

        self._save_checkpoint()
        return all_posts

    def fetch_latest(self) -> List[NormalizedPost]:
        """Alias for fetch_feeds."""
        return self.fetch_feeds()

    def yield_batches(self, batch_size: int = 50) -> Iterator[List[NormalizedPost]]:
        """Yields batches of parsed RSS posts."""
        posts = self.fetch_feeds()
        for i in range(0, len(posts), batch_size):
            yield posts[i:i + batch_size]

    def stream_batches(self, batch_size: int = 50) -> Iterator[List[NormalizedPost]]:
        """Alias for yield_batches."""
        yield from self.yield_batches(batch_size=batch_size)

