#!/usr/bin/env python3
"""
scripts/run_live_hn_rss_shadow.py
==================================
Production-Hardened Live Multi-Source Streaming & Shadow Logging Runner.

Ingests live stories from:
- Hacker News Firebase REST API (free, unauthenticated)
- Technology RSS / Atom Feeds (BBC, NYT)

Production Safeguards:
1. Rotating file log handler (prevents disk bloat on long runs)
2. State checkpointing (survives crashes/restarts without duplicating posts)
3. Exponential network backoff on API rate limits or connection drops
4. Periodic 5-minute telemetry snapshots written to logs/telemetry_snapshots.jsonl
5. Clean shutdown after target --hours / --duration_minutes or SIGINT/SIGTERM
"""

import sys
import time
import json
import signal
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.hackernews import HackerNewsDataSource
from src.ingestion.rss import RSSDataSource
from src.preprocessing import clean_reddit_text
from src.embeddings import IncrementalFAISSIndex
from src.shadow_mode import ShadowLogger
from src.telemetry import get_telemetry

logger = logging.getLogger("live_hn_rss_shadow")

CHECKPOINT_FILE = ROOT_DIR / "logs" / "live_shadow_checkpoint.json"
TELEMETRY_SNAPSHOT_FILE = ROOT_DIR / "logs" / "telemetry_snapshots.jsonl"
LOG_STREAM_FILE = ROOT_DIR / "logs" / "live_shadow_stream.log"
SHADOW_LOG_FILE = ROOT_DIR / "logs" / "shadow_live.jsonl"


def setup_logging(log_level: int = logging.INFO):
    """Configures console and rotating file logging."""
    LOG_STREAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # Rotating File Handler (10MB per file, max 5 backup logs)
    rfh = RotatingFileHandler(
        str(LOG_STREAM_FILE),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    rfh.setLevel(log_level)
    rfh.setFormatter(formatter)
    root_logger.addHandler(rfh)


def load_checkpoint() -> Dict[str, Any]:
    """Loads checkpoint state if present."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("Loaded existing checkpoint: %d seen post IDs, total processed: %d",
                            len(data.get("seen_ids", [])), data.get("total_processed", 0))
                return data
        except Exception as e:
            logger.warning("Failed to parse checkpoint file, starting fresh: %s", e)
    return {"seen_ids": [], "total_processed": 0, "session_count": 0}


def save_checkpoint(state: Dict[str, Any]):
    """Atomically saves checkpoint state."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CHECKPOINT_FILE.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp_path.replace(CHECKPOINT_FILE)
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)


def append_telemetry_snapshot(snapshot: Dict[str, Any]):
    """Appends a 5-minute telemetry snapshot to jsonl."""
    TELEMETRY_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(TELEMETRY_SNAPSHOT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot) + "\n")
    except Exception as e:
        logger.error("Failed to append telemetry snapshot: %s", e)


class LiveShadowRunner:
    """Manages resilient streaming ingestion with checkpointing, backoff, and telemetry."""

    def __init__(self, target_duration_seconds: float, poll_interval: float = 15.0):
        self.target_duration_seconds = target_duration_seconds
        self.poll_interval = poll_interval
        self.running = True
        
        # Signals for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Telemetry & Logger
        self.telemetry = get_telemetry()
        self.telemetry.reset()
        self.shadow_logger = ShadowLogger(log_path=SHADOW_LOG_FILE)
        self.faiss_index = IncrementalFAISSIndex(dim=384, max_vectors=10000)

        # Ingestion sources
        self.hn_source = HackerNewsDataSource(max_stories_per_fetch=25)
        self.rss_source = RSSDataSource(max_items_per_feed=25)

        # Load Checkpoint State
        ckpt = load_checkpoint()
        self.seen_ids: Set[str] = set(ckpt.get("seen_ids", []))
        self.total_processed: int = ckpt.get("total_processed", 0)
        self.session_count: int = ckpt.get("session_count", 0) + 1

        self.consecutive_errors = 0
        self.max_backoff = 60.0

    def _handle_signal(self, signum, frame):
        logger.info("Termination signal %d received. Gracefully stopping stream...", signum)
        self.running = False

    def run(self):
        start_time = time.time()
        end_time = start_time + self.target_duration_seconds
        last_telemetry_snapshot = start_time
        snapshot_interval = 300.0  # 5 minutes
        iteration = 1

        logger.info("=" * 70)
        logger.info(" STARTING 24H-READY LIVE HN / RSS STREAMING RUNNER")
        logger.info(" Target Duration : %.1f hours (%.1f minutes / %.1f seconds)",
                    self.target_duration_seconds / 3600.0,
                    self.target_duration_seconds / 60.0,
                    self.target_duration_seconds)
        logger.info(" Poll Interval   : %.1f seconds", self.poll_interval)
        logger.info(" Active Sources  : HackerNews (Firebase), Tech RSS Feeds")
        logger.info(" Checkpoint File : %s", CHECKPOINT_FILE)
        logger.info(" Telemetry File  : %s", TELEMETRY_SNAPSHOT_FILE)
        logger.info(" Rotating Log    : %s", LOG_STREAM_FILE)
        logger.info("=" * 70)

        while self.running and time.time() < end_time:
            iter_start = time.time()
            elapsed = iter_start - start_time
            iter_new_posts = 0
            has_error = False

            logger.info("--- Iteration %d | Elapsed: %.1fs / %.1fs | Seen IDs: %d ---",
                        iteration, elapsed, self.target_duration_seconds, len(self.seen_ids))

            # 1. Ingest Hacker News Stories
            try:
                hn_posts = self.hn_source.fetch_top_stories()
                new_hn = [p for p in hn_posts if p.id not in self.seen_ids]
                for p in new_hn:
                    t0 = time.perf_counter()
                    p.text = clean_reddit_text(p.text)
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    self.telemetry.record_post(source=p.source, status="processed", latency_ms=lat_ms, timestamp=p.datetime_utc)
                    self.seen_ids.add(p.id)
                    self.total_processed += 1
                    iter_new_posts += 1
                logger.info("HackerNews: %d fetched, %d new unique posts processed", len(hn_posts), len(new_hn))
            except Exception as e:
                logger.error("HackerNews fetch error: %s", e)
                self.telemetry.record_error("hn_fetch_error")
                has_error = True

            # 2. Ingest RSS Feed Articles
            try:
                rss_posts = self.rss_source.fetch_feeds()
                new_rss = [p for p in rss_posts if p.id not in self.seen_ids]
                for p in new_rss:
                    t0 = time.perf_counter()
                    p.text = clean_reddit_text(p.text)
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    self.telemetry.record_post(source=p.source, status="processed", latency_ms=lat_ms, timestamp=p.datetime_utc)
                    self.seen_ids.add(p.id)
                    self.total_processed += 1
                    iter_new_posts += 1
                logger.info("RSS Feeds: %d fetched, %d new unique posts processed", len(rss_posts), len(new_rss))
            except Exception as e:
                logger.error("RSS fetch error: %s", e)
                self.telemetry.record_error("rss_fetch_error")
                has_error = True

            # 3. Handle Exponential Network Backoff
            if has_error:
                self.consecutive_errors += 1
                backoff_time = min(self.max_backoff, self.poll_interval * (2 ** (self.consecutive_errors - 1)))
                logger.warning("Network issue encountered (Streak: %d). Backing off for %.1fs...",
                               self.consecutive_errors, backoff_time)
                current_sleep = backoff_time
            else:
                self.consecutive_errors = 0
                current_sleep = self.poll_interval

            # 4. Checkpoint State
            # Cap stored seen_ids to last 50,000 to prevent unbound memory growth
            seen_ids_list = list(self.seen_ids)[-50000:] if len(self.seen_ids) > 50000 else list(self.seen_ids)
            save_checkpoint({
                "seen_ids": seen_ids_list,
                "total_processed": self.total_processed,
                "session_count": self.session_count,
                "last_active_timestamp": time.time()
            })

            # 5. Periodic 5-Minute Telemetry Snapshot
            now = time.time()
            if (now - last_telemetry_snapshot) >= snapshot_interval:
                metrics = self.telemetry.get_metrics()
                snapshot = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                    "elapsed_seconds": round(now - start_time, 1),
                    "iteration": iteration,
                    "posts_received": metrics["posts_received"],
                    "posts_processed": metrics["posts_processed"],
                    "posts_rejected": metrics["posts_rejected"],
                    "latency_ms": round(metrics["latency_ms"], 4),
                    "memory_use_mb": round(metrics["memory_use_mb"], 1),
                    "errors": metrics["errors"],
                    "unique_seen_ids": len(self.seen_ids)
                }
                append_telemetry_snapshot(snapshot)
                last_telemetry_snapshot = now
                logger.info(">>> [5-MIN TELEMETRY SNAPSHOT] Processed: %d | Memory: %.1f MB | Latency: %.4f ms | Errors: %d <<<",
                            snapshot["posts_processed"], snapshot["memory_use_mb"], snapshot["latency_ms"], snapshot["errors"])

            iteration += 1
            remaining = end_time - time.time()
            if remaining <= 0:
                break

            sleep_duration = min(current_sleep, max(0.1, remaining))
            time.sleep(sleep_duration)

        # Final Shutdown Telemetry & Checkpoint
        final_time = time.time()
        final_metrics = self.telemetry.get_metrics()
        final_snapshot = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(final_time)),
            "elapsed_seconds": round(final_time - start_time, 1),
            "status": "completed" if (final_time >= end_time) else "stopped_by_user",
            "iteration": iteration,
            "posts_received": final_metrics["posts_received"],
            "posts_processed": final_metrics["posts_processed"],
            "posts_rejected": final_metrics["posts_rejected"],
            "latency_ms": round(final_metrics["latency_ms"], 4),
            "memory_use_mb": round(final_metrics["memory_use_mb"], 1),
            "errors": final_metrics["errors"],
            "unique_seen_ids": len(self.seen_ids)
        }
        append_telemetry_snapshot(final_snapshot)

        print("\n" + "=" * 75)
        print("               LIVE HN / RSS SHADOW SESSION SUMMARY")
        print("=" * 75)
        print(f"Total Elapsed Time : {final_time - start_time:.2f} seconds ({(final_time - start_time)/60:.2f} minutes)")
        print(f"Iterations Run     : {iteration}")
        print(f"Session Processed  : {final_metrics['posts_processed']}")
        print(f"Total Unique IDs   : {len(self.seen_ids)}")
        print(f"Processing Latency : {final_metrics['latency_ms']:.4f} ms")
        print(f"Process Memory RSS : {final_metrics['memory_use_mb']:.1f} MB")
        print(f"Errors Logged      : {final_metrics['errors']}")
        print(f"Checkpoint File    : {CHECKPOINT_FILE}")
        print(f"Telemetry Snapshot : {TELEMETRY_SNAPSHOT_FILE}")
        print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="24H-Ready Live HN/RSS Streamer & Shadow Runner")
    parser.add_argument("--hours", type=float, default=None, help="Run duration in hours (e.g. 24.0 or 0.166667)")
    parser.add_argument("--duration_minutes", type=float, default=None, help="Run duration in minutes (e.g. 10.0)")
    parser.add_argument("--duration_seconds", type=float, default=None, help="Run duration in seconds")
    parser.add_argument("--interval", type=float, default=15.0, help="Poll interval in seconds (default: 15.0s)")
    args = parser.parse_args()

    setup_logging()

    if args.hours is not None:
        duration_sec = args.hours * 3600.0
    elif args.duration_minutes is not None:
        duration_sec = args.duration_minutes * 60.0
    elif args.duration_seconds is not None:
        duration_sec = args.duration_seconds
    else:
        duration_sec = 600.0  # Default 10 minutes

    runner = LiveShadowRunner(target_duration_seconds=duration_sec, poll_interval=args.interval)
    runner.run()


if __name__ == "__main__":
    main()
