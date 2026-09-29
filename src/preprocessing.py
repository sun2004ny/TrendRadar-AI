import os
import re
import pandas as pd

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

def preprocess_and_window(
    input_path: str = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Step 6A Part 2: Preprocesses the 6-Week Multi-Week Reddit Dataset.
    1. Loads data/raw/reddit_sample_6weeks.csv.
    2. Cleans title and selftext into 'clean_text'.
    3. Filters out deleted, empty, or trivially short posts (< 15 chars, < 3 words).
    4. Strictly sorts chronologically (Oldest -> Newest) from July to September 2021.
    5. Preserves the 6 distinct weekly windows (W_0 to W_5).
    6. Saves to data/processed/reddit_cleaned_6weeks.csv.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if input_path is None:
        input_path = os.path.join(base_dir, "data", "raw", "reddit_sample_6weeks.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "processed", "reddit_cleaned_6weeks.csv")

    print(f"1. Loading 6-week raw data from: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"   Initial rows: {initial_count}")

    # Ensure timestamp is parsed properly
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df.dropna(subset=["created_utc"]).copy()
    df["datetime_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

    # 2. Text Cleaning: Combine title and selftext
    print("2. Cleaning text fields...")
    df["clean_title"] = df["title"].apply(clean_reddit_text)
    df["clean_selftext"] = df["selftext"].fillna("").apply(clean_reddit_text)
    
    # Combine title and selftext if selftext contains helpful context
    df["clean_text"] = df.apply(
        lambda r: f"{r['clean_title']}. {r['clean_selftext']}".strip(". ") 
        if len(r['clean_selftext']) > 5 else r['clean_title'],
        axis=1
    )

    # Filter out empty or trivially short text (< 15 characters or < 3 words)
    df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))
    df = df[df["clean_text"].str.len() >= 15].copy()
    df = df[df["word_count"] >= 3].copy()
    print(f"   Filtered out {initial_count - len(df)} trivial/empty posts. Remaining: {len(df)}")

    # 3. CRITICAL: Strictly sort chronologically from oldest to newest (July -> Sept)
    print("3. Sorting chronologically (Oldest -> Newest)...")
    df = df.sort_values("datetime_utc", ascending=True).reset_index(drop=True)

    # 4. Standardize Window IDs (W_0 to W_5)
    print("4. Mapping the 6 weekly checkpoints to sequential window IDs...")
    if "week_index" in df.columns:
        df["window_index"] = df["week_index"]
        df["window_id"] = df["week_index"].apply(lambda idx: f"W_{idx}")
    else:
        unique_weeks = sorted(df["week_label"].unique())
        week_map = {w: idx for idx, w in enumerate(unique_weeks)}
        df["window_index"] = df["week_label"].map(week_map)
        df["window_id"] = df["window_index"].apply(lambda idx: f"W_{idx}")

    # Select core columns to keep the file lean and clean
    core_columns = [
        "id", "created_utc", "datetime_utc", "week_label", "window_id", 
        "window_index", "clean_text", "word_count", "score"
    ]
    df_clean = df[core_columns].copy()

    # 5. Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"5. Saved cleaned multi-week dataset to: {output_path}")

    # Summary statistics
    print("\n--- 6-Week Preprocessing Summary ---")
    print(f"Total Cleaned Posts: {len(df_clean):,}")
    print(f"Overall Date Range: {df_clean['datetime_utc'].min()} to {df_clean['datetime_utc'].max()}")
    print("\nCleaned Posts per Weekly Window:")
    window_summary = df_clean.groupby(["window_index", "window_id", "week_label"]).agg(
        start_date=("datetime_utc", "min"),
        end_date=("datetime_utc", "max"),
        post_count=("id", "count")
    ).reset_index()
    print(window_summary.to_string(index=False))

    return df_clean

if __name__ == "__main__":
    preprocess_and_window()
