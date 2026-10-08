"""
src/baseline.py - Conventional Volume-Based Trend Detector (Baseline)
Uses pathlib, Python logging, and centralized config.
"""

from pathlib import Path
import re
import logging
from collections import Counter
from typing import Optional, Tuple
import pandas as pd

from src.config import ROOT_DIR, get_config

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "is", "are", "was", "were", "what", "how", "why",
    "who", "where", "which", "your", "you", "my", "i", "it", "this",
    "that", "do", "does", "did", "have", "has", "had", "can", "could",
    "would", "should", "if", "be", "been", "being", "from", "as", "by"
}


def extract_ngrams(text: str, n_range: Tuple[int, ...] = (1, 2)):
    """Extracts simple unigrams and bigrams from cleaned text."""
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower()) if w not in STOPWORDS]
    ngrams = []
    if 1 in n_range:
        ngrams.extend(words)
    if 2 in n_range:
        for i in range(len(words) - 1):
            ngrams.append(f"{words[i]} {words[i+1]}")
    return ngrams


def run_volume_baseline(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    min_freq: Optional[int] = None,
    surge_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Conventional Volume-Based Trend Detector (Baseline):
    Computes n-gram frequency surges window-by-window.
    """
    cfg = get_config()
    freq_min = min_freq if min_freq is not None else int(cfg.baseline.get("min_freq", 5))
    surge_thresh = surge_threshold if surge_threshold is not None else float(cfg.baseline.get("surge_threshold", 2.0))
    n_range = tuple(cfg.baseline.get("ngram_range", [1, 2]))

    if input_path is None:
        input_path = ROOT_DIR / "data" / "processed" / "reddit_cleaned_6weeks.csv"
    else:
        input_path = Path(input_path)

    if output_path is None:
        output_path = ROOT_DIR / "data" / "processed" / "baseline_volume_alerts_6weeks.csv"
    else:
        output_path = Path(output_path)

    logger.info("Loading processed data from: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Total posts to analyze: %d", len(df))

    windows = sorted(df["window_id"].unique(), key=lambda w: int(w.split("_")[1]))
    logger.info("Total windows to analyze: %d", len(windows))

    # Window-by-window n-gram frequency counter
    window_term_counts = {}
    for win in windows:
        win_df = df[df["window_id"] == win]
        term_counter = Counter()
        for text in win_df["clean_text"]:
            for ngram in extract_ngrams(text, n_range=n_range):
                term_counter[ngram] += 1
        window_term_counts[win] = term_counter

    alerts = []
    all_terms = set()
    for counts in window_term_counts.values():
        all_terms.update(counts.keys())

    logger.info("Detecting volume bursts across time windows (min_freq=%d, surge=%.1fx)...", freq_min, surge_thresh)
    for term in all_terms:
        history = []
        for t_idx, win in enumerate(windows):
            current_count = window_term_counts[win].get(term, 0)

            if t_idx > 0 and current_count >= freq_min:
                avg_past_volume = sum(history) / len(history) if history else 0

                if avg_past_volume > 0:
                    surge_ratio = current_count / avg_past_volume
                else:
                    surge_ratio = float(current_count)

                if surge_ratio >= surge_thresh:
                    alerts.append({
                        "term": term,
                        "detected_window": win,
                        "window_index": t_idx,
                        "current_volume": current_count,
                        "past_avg_volume": round(avg_past_volume, 2),
                        "surge_ratio": round(surge_ratio, 2)
                    })
                    break

            history.append(current_count)

    alert_df = pd.DataFrame(alerts)
    if not alert_df.empty:
        alert_df = alert_df.sort_values(by=["window_index", "surge_ratio"], ascending=[True, False]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    alert_df.to_csv(output_path, index=False)
    logger.info("Volume Baseline generated %d trend alerts to: %s", len(alert_df), output_path)

    return alert_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_volume_baseline()
