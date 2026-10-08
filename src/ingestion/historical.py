"""
src/ingestion/historical.py - Historical File-Based Ingestion Source

Loads historical datasets (CSV, daily files, or custom paths) through the NormalizedPost interface.
Rejects and tracks malformed records without crashing.
"""

from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any, Union
import logging
import pandas as pd

from src.config import ROOT_DIR
from src.ingestion.base import DataSource, NormalizedPost

logger = logging.getLogger(__name__)


class HistoricalDataSource(DataSource):
    """
    Ingests posts from historical CSV files into batches of NormalizedPost objects.
    Enforces validation: rejects missing timestamps, empty text, non-UTC times, and logs counts.
    """

    def __init__(
        self,
        file_path: Optional[Union[Path, str]] = None,
        source_name: str = "reddit_historical"
    ):
        super().__init__(source_name=source_name)
        if file_path is None:
            # Default to canonical processed daily dataset
            default_p = ROOT_DIR / "data" / "processed" / "daily" / "reddit_cleaned_daily.csv"
            raw_p = ROOT_DIR / "data" / "raw" / "reddit_daily_30days.csv"
            self.file_path = default_p if default_p.exists() else raw_p
        else:
            self.file_path = Path(file_path)

    def _extract_post_payload(self, row: pd.Series) -> Dict[str, Any]:
        """Maps DataFrame row values to NormalizedPost schema input fields."""
        raw_id = str(row.get("id", "")).strip()

        # Text selection: priority clean_text -> title + selftext -> title
        text = ""
        if "clean_text" in row and pd.notna(row["clean_text"]):
            text = str(row["clean_text"]).strip()
        elif "title" in row and pd.notna(row["title"]):
            title = str(row["title"]).strip()
            selftext = str(row.get("selftext", "")).strip() if pd.notna(row.get("selftext")) else ""
            if selftext and selftext not in ["[removed]", "[deleted]"]:
                text = f"{title}. {selftext}".strip(". ")
            else:
                text = title

        # Timestamp selection: priority datetime_utc -> created_utc
        timestamp_val = None
        if "datetime_utc" in row and pd.notna(row["datetime_utc"]):
            timestamp_val = row["datetime_utc"]
        elif "created_utc" in row and pd.notna(row["created_utc"]):
            timestamp_val = row["created_utc"]

        # Preserve extra metadata
        meta = {}
        for k in ["window_id", "window_index", "day_id", "day_index", "score", "week_label", "clean_text"]:
            if k in row and pd.notna(row[k]):
                meta[k] = row[k]

        return {
            "id": raw_id,
            "raw_id": raw_id,
            "text": text,
            "timestamp": timestamp_val,
            "source": self.source_name,
            "author": str(row.get("author", "unknown")),
            "metadata": meta
        }

    def yield_batches(self, batch_size: int = 100) -> Iterator[List[NormalizedPost]]:
        """
        Streams validated NormalizedPost instances in fixed-size batches from the underlying file.
        Skips invalid rows and increments invalid_posts_count.
        """
        if not self.file_path.exists():
            logger.error("Dataset file does not exist: %s", self.file_path)
            return

        logger.info("Ingesting historical data from %s (batch_size=%d)...", self.file_path, batch_size)
        current_batch: List[NormalizedPost] = []

        # Read CSV in chunks for memory efficiency
        for chunk in pd.read_csv(self.file_path, chunksize=max(batch_size, 500)):
            for _, row in chunk.iterrows():
                payload = self._extract_post_payload(row)
                post = self.validate_and_normalize(payload)
                if post is not None:
                    current_batch.append(post)
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []

        if current_batch:
            yield current_batch

        logger.info(
            "Completed ingestion from %s: %d valid posts, %d invalid rejected.",
            self.file_path,
            self.valid_posts_count,
            self.invalid_posts_count
        )
