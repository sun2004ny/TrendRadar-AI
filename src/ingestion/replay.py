"""
src/ingestion/replay.py - Historical Replay Data Source

Simulates a live streaming source by replaying the historical dataset day-by-day (checkpoint-by-checkpoint).
"""

from pathlib import Path
import sys
import logging
from typing import Iterator, List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.base import DataSource, NormalizedPost

logger = logging.getLogger(__name__)


class ReplayDataSource(DataSource):
    """
    Feeds historical daily checkpoints sequentially, simulating live real-time ingestion.
    """

    def __init__(
        self,
        daily_dir: Optional[Path] = None,
        start_day_idx: int = 1,
        max_days: Optional[int] = None
    ):
        super().__init__(source_name="replay")
        self.daily_dir = Path(daily_dir) if daily_dir else ROOT_DIR / "data" / "raw" / "daily"
        self.start_day_idx = start_day_idx
        self.max_days = max_days
        self._current_day_idx = start_day_idx

        # Discover all available checkpoint files
        if self.daily_dir.exists():
            self._files = sorted(
                list(self.daily_dir.glob("D_*.csv")),
                key=lambda f: int(f.stem.split("_")[1]) if "_" in f.stem else 0
            )
        else:
            self._files = []

        if max_days:
            self._files = self._files[:max_days]

    def reset(self) -> None:
        """Resets replay cursor back to start."""
        self._current_day_idx = self.start_day_idx

    def stream_checkpoints(self) -> Iterator[Tuple[str, List[NormalizedPost]]]:
        """
        Yields posts grouped by daily checkpoint: (checkpoint_id, [NormalizedPost, ...]).
        """
        for f in self._files:
            ckpt_id = f.stem
            d_idx = int(ckpt_id.split("_")[1]) if "_" in ckpt_id else 0
            if d_idx < self.start_day_idx:
                continue
            if self.max_days and (d_idx - self.start_day_idx >= self.max_days):
                break

            posts = self._load_file(f, ckpt_id, d_idx)
            yield ckpt_id, posts

    def stream_batches(self, batch_size: int = 100) -> Iterator[List[NormalizedPost]]:
        """Yields continuous batches of posts across checkpoints."""
        batch: List[NormalizedPost] = []
        for _, posts in self.stream_checkpoints():
            for p in posts:
                batch.append(p)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def yield_batches(self, batch_size: int = 100) -> Iterator[List[NormalizedPost]]:
        """Implementation of DataSource.yield_batches."""
        yield from self.stream_batches(batch_size=batch_size)


    def load_all(self) -> List[NormalizedPost]:
        """Loads all posts across all replay checkpoints."""
        all_posts: List[NormalizedPost] = []
        for _, posts in self.stream_checkpoints():
            all_posts.extend(posts)
        return all_posts

    def _load_file(self, file_path: Path, ckpt_id: str, d_idx: int) -> List[NormalizedPost]:
        posts: List[NormalizedPost] = []
        try:
            df = pd.read_csv(file_path)
            for _, r in df.iterrows():
                try:
                    pid = str(r.get("id", "")).strip()
                    title = str(r.get("title", ""))
                    selftext = str(r.get("selftext", ""))
                    text = f"{title} {selftext}".strip()
                    if not text or text == "None" or text == "nan":
                        text = title or selftext or ""

                    ts_val = r.get("created_utc", 0)
                    meta = {
                        "window_id": ckpt_id,
                        "day_id": ckpt_id,
                        "day_index": d_idx,
                        "score": r.get("score", 0),
                        "num_comments": r.get("num_comments", 0)
                    }

                    p = NormalizedPost(
                        id=pid,
                        raw_id=pid,
                        text=text,
                        timestamp=ts_val,
                        source="reddit_replay",
                        metadata=meta
                    )
                    posts.append(p)
                    self.valid_posts_count += 1
                except Exception as e:
                    self.invalid_posts_count += 1
                    logger.debug("Rejected invalid post: %s", e)
        except Exception as e:
            logger.error("Error reading replay file %s: %s", file_path, e)
        return posts

