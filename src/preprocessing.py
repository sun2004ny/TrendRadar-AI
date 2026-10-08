"""
src/preprocessing.py - Text cleaning, question-template stripping, and chronological windowing
Uses pathlib, Python logging, and NormalizedPost objects.
"""

from pathlib import Path
import re
import logging
from typing import Optional, List, Dict, Any, Union
import pandas as pd

from src.ingestion.base import NormalizedPost

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


def strip_interrogative_templates(text: str) -> str:
    """
    Strips leading interrogative and Reddit conversational question phrasing
    so clustering groups by topical substance instead of grammatical shell.
    """
    if not isinstance(text, str):
        return ""

    patterns = [
        r"^(what|how|why|who|where|when)(\s+is|\s+are|\s+do|\s+does|\s+did|\s+was|\s+were|\s+would|\s+could|\s+should)?\b",
        r"^(people|redditors|men|women|doctors|teachers|users)\s+of\s+reddit\b",
        r"^(what\s+is|what\'s)\s+(the\s+most|your\s+favorite|something|a|an)\b",
        r"^(if\s+you\s+could|have\s+you\s+ever)\b",
    ]
    cleaned = text.strip()
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def clean_reddit_text(text: str) -> str:
    """
    Cleans raw Reddit text:
    - Strips URLs
    - Normalizes extra whitespace and newlines
    - Removes common Reddit markers like [deleted] and [removed]
    """
    if not isinstance(text, str):
        return ""

    # Remove boilerplate removal tags
    if text.strip().lower() in ["[deleted]", "[removed]"]:
        return ""

    # Remove URLs (http/https links)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML entity artifacts (like &amp;, &lt;, &gt;)
    text = re.sub(r'&\w+;', ' ', text)

    # Remove multiple spaces / newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_normalized_post(post: NormalizedPost) -> Optional[NormalizedPost]:
    """
    Applies standard cleaning and question-template stripping directly to a NormalizedPost.
    Drops posts that are trivially short (< 15 characters or < 3 words).
    Returns a new NormalizedPost with cleaned text, or None if filtered out.
    """
    cleaned = clean_reddit_text(post.text)
    cleaned = strip_interrogative_templates(cleaned)
    words = cleaned.split()
    if len(cleaned) < 15 or len(words) < 3:
        return None

    new_meta = dict(post.metadata)
    new_meta["clean_text"] = cleaned
    new_meta["word_count"] = len(words)

    return NormalizedPost(
        id=post.id,
        raw_id=post.raw_id,
        text=cleaned,
        timestamp=post.timestamp,
        source=post.source,
        author=post.author,
        metadata=new_meta
    )


def preprocess_normalized_posts(posts: List[NormalizedPost]) -> List[NormalizedPost]:
    """
    Transforms a sequence of NormalizedPost instances into a chronologically sorted,
    cleaned sequence. Operates purely on NormalizedPost attributes, without dataframe columns.
    """
    cleaned_posts: List[NormalizedPost] = []
    for p in posts:
        cp = clean_normalized_post(p)
        if cp is not None:
            cleaned_posts.append(cp)

    # Chronologically sort oldest to newest
    cleaned_posts.sort(key=lambda p: p.timestamp)
    return cleaned_posts


def preprocess_and_window(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Preprocesses the Multi-Week Reddit Dataset:
    1. Loads raw csv.
    2. Cleans text and strips templates.
    3. Filters out deleted, empty, or short posts (< 15 chars, < 3 words).
    4. Chronologically sorts (Oldest -> Newest).
    5. Preserves sequential window IDs.
    """
    if input_path is None:
        input_path = BASE_DIR / "data" / "raw" / "reddit_sample_6weeks.csv"
    else:
        input_path = Path(input_path)

    if output_path is None:
        output_path = BASE_DIR / "data" / "processed" / "reddit_cleaned_6weeks.csv"
    else:
        output_path = Path(output_path)

    logger.info("Loading raw data from: %s", input_path)
    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info("Initial rows: %d", initial_count)

    # Ensure timestamp is parsed properly
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df.dropna(subset=["created_utc"]).copy()
    df["datetime_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

    # Text Cleaning: Combine title and selftext
    logger.info("Cleaning text fields and stripping interrogative templates...")
    df["clean_title"] = df["title"].apply(clean_reddit_text)
    df["clean_selftext"] = df["selftext"].fillna("").apply(clean_reddit_text)

    # Combine title and selftext if selftext contains helpful context
    df["clean_text"] = df.apply(
        lambda r: f"{r['clean_title']}. {r['clean_selftext']}".strip(". ")
        if len(r['clean_selftext']) > 5 else r['clean_title'],
        axis=1
    )

    # Apply template stripping
    df["clean_text"] = df["clean_text"].apply(strip_interrogative_templates)

    # Filter out empty or trivially short text (< 15 characters or < 3 words)
    df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))
    df = df[df["clean_text"].str.len() >= 15].copy()
    df = df[df["word_count"] >= 3].copy()
    logger.info("Filtered out %d trivial/empty posts. Remaining: %d", initial_count - len(df), len(df))

    # Chronologically sort from oldest to newest
    logger.info("Sorting chronologically (Oldest -> Newest)...")
    df = df.sort_values("datetime_utc", ascending=True).reset_index(drop=True)

    # Standardize Window IDs (W_0 to W_5)
    logger.info("Mapping checkpoints to sequential window IDs...")
    if "week_index" in df.columns:
        df["window_index"] = df["week_index"]
        df["window_id"] = df["week_index"].apply(lambda idx: f"W_{idx}")
    else:
        unique_weeks = sorted(df["week_label"].unique())
        week_map = {w: idx for idx, w in enumerate(unique_weeks)}
        df["window_index"] = df["week_label"].map(week_map)
        df["window_id"] = df["window_index"].apply(lambda idx: f"W_{idx}")

    core_columns = [
        "id", "created_utc", "datetime_utc", "week_label", "window_id",
        "window_index", "clean_text", "word_count", "score"
    ]
    df_clean = df[core_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset to: %s", output_path)

    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    preprocess_and_window()
