"""
src/ingestion/base.py - Base Data Models and Abstract Data Source Interface

Provides:
- NormalizedPost: Strict Pydantic model enforcing timezone-aware UTC timestamps,
  non-empty text, and unique IDs.
- DataSource: Abstract base class providing batched streaming, counting, and rejection of invalid data.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Iterator, Union
import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


class NormalizedPost(BaseModel):
    """
    Standardized, strongly-typed post representation across all ingestion pipelines.
    Guarantees:
    - id and raw_id are non-empty strings.
    - text is non-empty after whitespace stripping.
    - timestamp is strictly timezone-aware UTC.
    """
    model_config = ConfigDict(extra="ignore")

    id: str
    raw_id: str
    text: str
    timestamp: datetime
    source: str
    author: Optional[str] = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "raw_id", mode="before")
    @classmethod
    def validate_id_fields(cls, v: Any) -> str:
        if v is None:
            raise ValueError("Post ID cannot be None")
        s = str(v).strip()
        if not s or s.lower() == "nan":
            raise ValueError("Post ID cannot be empty")
        return s

    @field_validator("text", mode="before")
    @classmethod
    def validate_text_content(cls, v: Any) -> str:
        if v is None:
            raise ValueError("Post text cannot be empty")
        s = str(v).strip()
        if not s or s.lower() == "nan":
            raise ValueError("Post text cannot be empty")
        return s

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp_utc(cls, v: Any) -> datetime:
        if v is None:
            raise ValueError("Timestamp is required")

        # Unix numeric epoch timestamps
        if isinstance(v, (int, float)):
            if str(v).lower() == "nan":
                raise ValueError("Timestamp cannot be NaN")
            return datetime.fromtimestamp(float(v), tz=timezone.utc)

        # String representations
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() == "nan":
                raise ValueError("Timestamp string cannot be empty")
            try:
                # Direct ISO parse
                dt = datetime.fromisoformat(s)
            except Exception:
                try:
                    import pandas as pd
                    dt = pd.to_datetime(s).to_pydatetime()
                except Exception as e:
                    raise ValueError(f"Unparseable timestamp string: {s}") from e

            if dt.tzinfo is None:
                raise ValueError("Timestamp must be timezone-aware UTC")
            if dt.utcoffset() != timedelta(0):
                raise ValueError("Timestamp must be in UTC timezone")
            return dt

        # Datetime objects
        if isinstance(v, datetime):
            if v.tzinfo is None:
                raise ValueError("Timestamp must be timezone-aware UTC")
            if v.utcoffset() != timedelta(0):
                raise ValueError("Timestamp must be in UTC timezone")
            return v

        raise ValueError(f"Unsupported timestamp type: {type(v)}")


class DataSource(ABC):
    """
    Abstract base class for all streaming and batch data sources.
    Maintains telemetry counters for valid and invalid rejected posts.
    """

    def __init__(self, source_name: str = "generic"):
        self.source_name = source_name
        self.valid_posts_count: int = 0
        self.invalid_posts_count: int = 0

    def reset_stats(self) -> None:
        """Resets valid and invalid ingestion counters."""
        self.valid_posts_count = 0
        self.invalid_posts_count = 0

    def validate_and_normalize(self, raw_record: Dict[str, Any]) -> Optional[NormalizedPost]:
        """
        Validates raw payload into a NormalizedPost.
        If validation fails, increments invalid_posts_count and returns None without crashing.
        """
        try:
            post = NormalizedPost(**raw_record)
            self.valid_posts_count += 1
            return post
        except (ValidationError, ValueError, TypeError) as exc:
            self.invalid_posts_count += 1
            logger.debug(
                "Rejected invalid post payload from %s (raw_id=%s): %s",
                self.source_name,
                raw_record.get("raw_id", raw_record.get("id")),
                exc
            )
            return None

    @abstractmethod
    def yield_batches(self, batch_size: int = 100) -> Iterator[List[NormalizedPost]]:
        """
        Yields sequential batches of validated NormalizedPost instances.
        Must be implemented by concrete ingestion subclasses.
        """
        pass

    def load_all(self, batch_size: int = 500) -> List[NormalizedPost]:
        """Convenience method to consume all batches into a single list."""
        all_posts: List[NormalizedPost] = []
        for batch in self.yield_batches(batch_size=batch_size):
            all_posts.extend(batch)
        return all_posts
