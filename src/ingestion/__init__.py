"""
src/ingestion - Data Source & Post Ingestion Layer
Normalized ingestion models and data source interfaces for TrendRadar AI.
"""

from src.ingestion.base import NormalizedPost, DataSource
from src.ingestion.historical import HistoricalDataSource
from src.ingestion.reddit import RedditLiveDataSource, DailyRequestBudgetLimiter

__all__ = ["NormalizedPost", "DataSource", "HistoricalDataSource", "RedditLiveDataSource", "DailyRequestBudgetLimiter"]
