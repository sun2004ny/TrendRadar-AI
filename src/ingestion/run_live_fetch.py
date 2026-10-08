"""
src/ingestion/run_live_fetch.py - Live Reddit Ingestion Fetch Runner

Executes a live fetch session (e.g. 10 minutes) across specified subreddits:
- Monitors posts received, rejected, and network/auth errors.
- Respects daily rate-limit budget and exponential backoff.
- Records last seen timestamps into persistent checkpoint.
- Updates the incremental FAISS vector index.
"""

from pathlib import Path
import sys
import time
import argparse
import logging
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.reddit import RedditLiveDataSource

logger = logging.getLogger("live_ingestion_runner")


def run_live_fetch(
    duration_minutes: float = 10.0,
    subreddits: list = None,
    poll_interval_seconds: float = 30.0
):
    subreddits = subreddits or ["AskReddit", "technology"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 75)
    print("      TRENDRADAR-AI: LIVE REDDIT INGESTION SESSION (PHASE 5)")
    print("=" * 75)
    print(f"Target Subreddits : {', '.join(f'r/{s}' for s in subreddits)}")
    print(f"Duration Target   : {duration_minutes:.1f} minutes")
    print(f"Polling Interval  : {poll_interval_seconds:.1f} seconds")
    print("=" * 75 + "\n")

    ds = RedditLiveDataSource(subreddits=subreddits)

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    total_received = 0
    total_rejected = 0
    errors_logged = []

    if not ds.is_authenticated:
        error_msg = (
            "No active Reddit API credentials found in .env (or credentials are placeholder). "
            "Live ingestion cannot establish an authenticated OAuth connection to Reddit API."
        )
        logger.warning(error_msg)
        errors_logged.append(error_msg)
    else:
        logger.info("Authenticated PRAW connection established. Beginning live ingestion loop...")
        iteration = 1
        while time.time() < end_time:
            logger.info("--- Polling Cycle %d (Elapsed: %.1fs) ---", iteration, time.time() - start_time)
            try:
                batches = list(ds.yield_batches(batch_size=20))
                for batch in batches:
                    total_received += len(batch)
                    logger.info("Batch yielded: %d posts (checkpoint ts=%.1f)", len(batch), ds.last_seen_timestamp)
                total_rejected = ds.invalid_posts_count
            except Exception as e:
                err_str = f"Error during polling cycle {iteration}: {e}"
                logger.error(err_str)
                errors_logged.append(err_str)

            iteration += 1
            remaining = end_time - time.time()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_seconds, remaining))

    elapsed_actual = time.time() - start_time

    print("\n" + "=" * 75)
    print("                   LIVE FETCH SESSION SUMMARY")
    print("=" * 75)
    print(f"Status              : {'Completed (OAuth Active)' if ds.is_authenticated else 'Halted (Awaiting .env Credentials)'}")
    print(f"Elapsed Time        : {elapsed_actual:.2f}s ({elapsed_actual/60:.2f} min)")
    print(f"Subreddits Polled   : {', '.join(f'r/{s}' for s in subreddits)}")
    print(f"Posts Received      : {total_received}")
    print(f"Posts Rejected      : {total_rejected}")
    print(f"Errors Encountered  : {len(errors_logged)}")
    if errors_logged:
        print("\nErrors / Warnings:")
        for idx, err in enumerate(errors_logged, 1):
            print(f"  {idx}. {err}")
    print("=" * 75 + "\n")

    return {
        "is_authenticated": ds.is_authenticated,
        "elapsed_seconds": elapsed_actual,
        "posts_received": total_received,
        "posts_rejected": total_rejected,
        "errors": errors_logged
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Reddit Ingestion Runner")
    parser.add_argument("--duration", type=float, default=0.1, help="Duration in minutes (e.g. 10.0)")
    parser.add_argument("--subreddits", type=str, default="AskReddit,technology", help="Comma-separated subreddits")
    args = parser.parse_args()

    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    run_live_fetch(duration_minutes=args.duration, subreddits=subs)
