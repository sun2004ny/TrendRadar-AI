#!/usr/bin/env python3
"""
scripts/summarize_shadow.py
===========================
Parses and summarizes the shadow mode execution logs (logs/shadow_log.jsonl).
Reports topic candidates evaluated, duplicates dropped, latency distribution,
and memory usage over time.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.telemetry import get_telemetry


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize TrendRadar-AI Shadow Mode Logs")
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(ROOT_DIR / "logs" / "shadow_log.jsonl"),
        help="Path to shadow mode JSONL log file (default: logs/shadow_log.jsonl)"
    )
    return parser.parse_args()


def summarize_shadow_logs(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        print(f"[!] Shadow log file not found at: {log_path}")
        return {"total_records": 0}

    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print(f"[!] No valid JSON records found in {log_path}")
        return {"total_records": 0}

    total_records = len(records)
    unique_trajectories = set()
    duplicates_seen = 0
    day_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    coherences: List[float] = []
    emergence_scores: List[float] = []

    for r in records:
        tid = r.get("trajectory_id", "UNKNOWN")
        if tid in unique_trajectories:
            duplicates_seen += 1
        else:
            unique_trajectories.add(tid)

        day = r.get("day", "UNKNOWN")
        day_counts[day] = day_counts.get(day, 0) + 1

        status = r.get("novelty_status", "passed")
        status_counts[status] = status_counts.get(status, 0) + 1

        if "coherence" in r and r["coherence"] is not None:
            coherences.append(float(r["coherence"]))
        if "emergence_score" in r and r["emergence_score"] is not None:
            emergence_scores.append(float(r["emergence_score"]))

    telemetry = get_telemetry().get_metrics()

    print("\n" + "=" * 65)
    print("        TRENDRADAR-AI: SHADOW MODE EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Log File Path          : {log_path}")
    print(f"Total Candidate Alerts : {total_records}")
    print(f"Unique Trajectories    : {len(unique_trajectories)}")
    print(f"Duplicate Evals Dropped: {duplicates_seen}")
    print(f"Ingestion In Error     : {telemetry.get('errors', 0)}")
    print(f"Active Process RSS     : {telemetry.get('memory_use_mb', 0.0):.1f} MB")
    print(f"Vector Index Size      : {telemetry.get('vector_index_size', 0)} vectors")
    print("-" * 65)

    print("\n[Novelty Gate & Quarantine Breakdown]")
    for st, count in status_counts.items():
        pct = (count / total_records) * 100
        print(f"  - {st.upper():<15}: {count:4d} ({pct:5.1f}%)")

    print("\n[Score Distributions]")
    if coherences:
        print(f"  Coherence       : min={np.min(coherences):.3f}, mean={np.mean(coherences):.3f}, max={np.max(coherences):.3f}")
    if emergence_scores:
        print(f"  Emergence Score : min={np.min(emergence_scores):.3f}, mean={np.mean(emergence_scores):.3f}, max={np.max(emergence_scores):.3f}")

    print("\n[Top Windows Evaluated]")
    for day, count in sorted(day_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  - {day:<10}: {count} candidates")

    print("=" * 65 + "\n")

    return {
        "total_records": total_records,
        "unique_trajectories": len(unique_trajectories),
        "duplicates_dropped": duplicates_seen,
        "status_counts": status_counts,
        "memory_use_mb": telemetry.get("memory_use_mb", 0.0)
    }


def main():
    args = parse_args()
    summarize_shadow_logs(Path(args.log_path))


if __name__ == "__main__":
    main()
