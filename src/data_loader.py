import os
import time
import urllib.request
import urllib.error
import json
import datetime
import pandas as pd

# 6 distinct checkpoints across the 1,000,000 Reddit rows:
# Spanning July, August, and September 2021 (~2.5 months)
WEEK_CHECKPOINTS = [
    (750000, "W0_Late_July"),
    (600000, "W1_Early_Aug"),
    (450000, "W2_Mid_Aug"),
    (300000, "W3_Early_Sep"),
    (150000, "W4_Mid_Sep"),
    (0,      "W5_Late_Sep"),
]

def fetch_batch_with_retry(url: str, max_retries: int = 5, base_delay: float = 2.0) -> list:
    """
    Fetches a batch of rows with automatic retry and exponential backoff
    specifically designed to handle HTTP 429 (Too Many Requests).
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [item["row"] for item in data.get("rows", [])]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = base_delay * (2 ** attempt)
                print(f"      [HTTP 429 Rate Limit] Backing off for {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"      [HTTP Error {e.code}]: {e}")
                time.sleep(2.0)
        except Exception as e:
            print(f"      [Network Error]: {e}")
            time.sleep(2.0)
            
    return []

def fetch_multiweek_sample(posts_per_week: int = 1000, output_path: str = None) -> pd.DataFrame:
    """
    Samples exactly posts_per_week across 6 distinct weeks spanning July to September 2021.
    Includes rate-limit backoff, polite request pacing, and resume support.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "raw", "reddit_sample_6weeks.csv")
    
    temp_dir = os.path.join(base_dir, "data", "raw", "temp_weeks")
    os.makedirs(temp_dir, exist_ok=True)

    batch_size = 100
    batches_per_week = max(1, posts_per_week // batch_size)
    all_week_dfs = []

    print(f"Fetching 6 distinct weeks ({posts_per_week} posts/week, total {posts_per_week*6} posts)...")

    for w_idx, (base_offset, week_name) in enumerate(WEEK_CHECKPOINTS):
        week_cache_file = os.path.join(temp_dir, f"{week_name}.csv")
        
        # Check if this week was already fully downloaded
        if os.path.exists(week_cache_file):
            existing_df = pd.read_csv(week_cache_file)
            if len(existing_df) >= posts_per_week:
                print(f"  [Cached] {week_name}: {len(existing_df)} posts already saved.")
                all_week_dfs.append(existing_df.head(posts_per_week))
                continue

        # If W5 (Late Sep), we can reuse the first 1,000 rows from reddit_sample_5000.csv directly!
        sample_5k_path = os.path.join(base_dir, "data", "raw", "reddit_sample_5000.csv")
        if week_name == "W5_Late_Sep" and os.path.exists(sample_5k_path):
            print(f"  [Local Instant Load] Loading {week_name} from local reddit_sample_5000.csv...")
            s5k_df = pd.read_csv(sample_5k_path).head(posts_per_week).copy()
            s5k_df["week_index"] = w_idx
            s5k_df["week_label"] = week_name
            s5k_df.to_csv(week_cache_file, index=False)
            all_week_dfs.append(s5k_df)
            continue

        print(f"  Fetching {week_name} (offset {base_offset})...")
        week_records = []
        
        for b in range(batches_per_week):
            cur_offset = base_offset + (b * batch_size)
            url = (
                f"https://datasets-server.huggingface.co/rows?dataset=SocialGrep/one-million-reddit-questions"
                f"&config=default&split=train&offset={cur_offset}&length={batch_size}"
            )
            
            # Polite delay between requests to avoid triggering rate limit
            time.sleep(0.7)
            batch = fetch_batch_with_retry(url)
            
            for r in batch:
                r["week_index"] = w_idx
                r["week_label"] = week_name
                week_records.append(r)
                
            print(f"    Progress: {len(week_records)}/{posts_per_week} posts collected")
            if len(week_records) >= posts_per_week:
                break

        week_df = pd.DataFrame(week_records[:posts_per_week])
        week_df.to_csv(week_cache_file, index=False)
        all_week_dfs.append(week_df)

    # Combine all 6 weeks
    final_df = pd.concat(all_week_dfs, ignore_index=True)
    final_df["created_utc"] = pd.to_numeric(final_df["created_utc"], errors="coerce")
    final_df["datetime_utc"] = pd.to_datetime(final_df["created_utc"], unit="s", utc=True)
    final_df["score"] = pd.to_numeric(final_df["score"], errors="coerce").fillna(0).astype(int)

    # CRITICAL: Sort strictly chronologically from oldest to newest
    final_df = final_df.sort_values("datetime_utc", ascending=True).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nSUCCESS: Exactly {len(final_df):,} posts across 6 weeks saved to:\n  {output_path}")

    # Summary
    print("\n--- Verified Multi-Week Dataset Summary ---")
    print(f"Total Posts: {len(final_df):,}")
    print(f"Overall Date Range: {final_df['datetime_utc'].min()} to {final_df['datetime_utc'].max()}")
    summary = final_df.groupby(["week_index", "week_label"]).agg(
        start_date=("datetime_utc", "min"),
        end_date=("datetime_utc", "max"),
        post_count=("id", "count")
    ).reset_index()
    print(summary.to_string(index=False))

    return final_df

if __name__ == "__main__":
    fetch_multiweek_sample(posts_per_week=1000)
