import os
import sys
import time
import urllib.request
import urllib.error
import json
import datetime
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def fetch_batch_with_retry(url: str, max_retries: int = 5, base_delay: float = 2.0) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [item["row"] for item in data.get("rows", [])]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = base_delay * (2 ** attempt)
                print(f"      [HTTP 429 Rate Limit] Backing off for {wait_time:.1f}s (retry {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"      [HTTP Error {e.code}]: {e}")
                time.sleep(2.0)
        except Exception as e:
            print(f"      [Network Error]: {e}")
            time.sleep(2.0)
    return []

def build_daily_dataset(
    num_days: int = 30,
    posts_per_day: int = 250,
    start_offset: int = 550000,
    step_offset: int = 10000,
    raw_output_path: str = None,
    cleaned_output_path: str = None
) -> pd.DataFrame:
    """
    Experiment 9A: Builds a 30-Day Fine-Grained Temporal Stream (D01 -> D30)
    
    Spans August 11, 2021 to September 9, 2021.
    Includes the critical pre-incubation, explosion, and decay phases of:
    - OnlyFans porn ban announcement (Aug 18 pre-incubation -> Aug 19 surge)
    - FDA Pfizer COVID Vaccine full approval (Aug 22 pre-incubation -> Aug 23 surge)
    - Reddit comment lock incident (Late Aug / Early Sep)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raw_output_path is None:
        raw_output_path = os.path.join(base_dir, "data", "raw", "reddit_daily_30days.csv")
    if cleaned_output_path is None:
        cleaned_output_path = os.path.join(base_dir, "data", "processed", "daily", "reddit_cleaned_daily.csv")

    cache_dir = os.path.join(base_dir, "data", "raw", "daily")
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 80)
    print(f"   STEP 9A: BUILDING {num_days}-DAY FINE-GRAINED TEMPORAL STREAM")
    print(f"   Target: {num_days} Daily Checkpoints x {posts_per_day} Posts/Day (~{num_days*posts_per_day} Posts)")
    print("=" * 80)

    daily_dfs = []

    for day_idx in range(num_days):
        day_num = day_idx + 1
        day_id = f"D_{day_num:02d}"
        day_cache_file = os.path.join(cache_dir, f"{day_id}.csv")
        day_offset = start_offset - (day_idx * step_offset)

        if os.path.exists(day_cache_file):
            print(f"   [{day_id}] Loading cached data from: {os.path.basename(day_cache_file)}")
            d_df = pd.read_csv(day_cache_file)
            daily_dfs.append(d_df)
            continue

        print(f"   [{day_id}] Fetching {posts_per_day} posts at offset {day_offset:,}...")
        rows_collected = []
        batch_size = 100
        needed_batches = (posts_per_day + batch_size - 1) // batch_size

        for b_idx in range(needed_batches):
            b_offset = day_offset + (b_idx * batch_size)
            limit = min(batch_size, posts_per_day - len(rows_collected))
            url = f"https://datasets-server.huggingface.co/rows?dataset=SocialGrep%2Fone-million-reddit-questions&config=default&split=train&offset={b_offset}&limit={limit}"
            
            batch_rows = fetch_batch_with_retry(url)
            if not batch_rows:
                print(f"      [Warning] No rows returned for offset {b_offset}. Pausing...")
                time.sleep(1.0)
                continue

            rows_collected.extend(batch_rows)
            time.sleep(0.3)  # Polite pacing

        if rows_collected:
            d_df = pd.DataFrame(rows_collected)
            d_df["day_id"] = day_id
            d_df["day_index"] = day_idx
            
            # Format datetime
            if "created_utc" in d_df.columns:
                d_df["datetime_utc"] = pd.to_datetime(d_df["created_utc"], unit="s", utc=True)
                sample_date = d_df["datetime_utc"].iloc[0].strftime("%Y-%m-%d")
                d_df["date_label"] = sample_date
                print(f"      Successfully fetched {len(d_df)} posts for {sample_date} ({day_id})")
            
            d_df.to_csv(day_cache_file, index=False)
            daily_dfs.append(d_df)
        else:
            print(f"      [Error] Failed to collect rows for {day_id}")

    if not daily_dfs:
        raise RuntimeError("No daily data collected!")

    all_raw = pd.concat(daily_dfs, ignore_index=True)
    all_raw.to_csv(raw_output_path, index=False)
    print(f"\n1. Saved combined raw daily dataset ({len(all_raw)} rows) to: {raw_output_path}")

    # Preprocessing & Cleaning
    print("2. Preprocessing & cleaning text fields...")
    cleaned_rows = []
    for _, row in all_raw.iterrows():
        title = str(row.get("title", ""))
        selftext = str(row.get("selftext", ""))
        full_text = f"{title} {selftext}".strip()
        
        # Remove deleted/removed
        if full_text in ["[deleted]", "[removed]", ""] or len(full_text.split()) < 3:
            continue
            
        cleaned_rows.append({
            "id": row.get("id", ""),
            "created_utc": row.get("created_utc", 0),
            "datetime_utc": row.get("datetime_utc", ""),
            "day_id": row.get("day_id", ""),
            "day_index": int(row.get("day_index", 0)),
            "date_label": row.get("date_label", ""),
            "clean_text": full_text,
            "word_count": len(full_text.split()),
            "score": row.get("score", 0)
        })

    clean_df = pd.DataFrame(cleaned_rows)
    clean_df = clean_df.sort_values(by="created_utc").reset_index(drop=True)
    
    os.makedirs(os.path.dirname(cleaned_output_path), exist_ok=True)
    clean_df.to_csv(cleaned_output_path, index=False)
    print(f"3. Saved cleaned daily dataset ({len(clean_df)} posts) to: {cleaned_output_path}")

    # Summary
    print("\n--- Daily Stream Summary ---")
    summary = clean_df.groupby(["day_index", "day_id", "date_label"]).agg(
        post_count=("id", "count"),
        start_time=("datetime_utc", "min"),
        end_time=("datetime_utc", "max")
    ).reset_index()
    print(summary.to_string())

    return clean_df

if __name__ == "__main__":
    build_daily_dataset()
