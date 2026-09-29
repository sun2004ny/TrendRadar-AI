import os
import re
from collections import Counter
import pandas as pd

def extract_ngrams(text: str, n_range=(1, 2)):
    """
    Extracts simple unigrams and bigrams from cleaned text,
    ignoring common English stop words.
    """
    # Basic English stopwords
    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
        "and", "or", "is", "are", "was", "were", "what", "how", "why",
        "who", "where", "which", "your", "you", "my", "i", "it", "this",
        "that", "do", "does", "did", "have", "has", "had", "can", "could",
        "would", "should", "if", "be", "been", "being", "from", "as", "by"
    }
    
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()) if w not in stopwords]
    ngrams = []
    
    # Unigrams (single words)
    if 1 in n_range:
        ngrams.extend(words)
        
    # Bigrams (two-word phrases)
    if 2 in n_range:
        for i in range(len(words) - 1):
            ngrams.append(f"{words[i]} {words[i+1]}")
            
    return ngrams

def run_volume_baseline(
    input_path: str = None,
    output_path: str = None,
    min_freq: int = 5,
    surge_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Conventional Volume-Based Trend Detector (Baseline):
    - Computes term/n-gram frequency for each chronological time window W_0, W_1, ...
    - Tracks raw volume of each phrase.
    - Flags a phrase as 'trending' when its volume in window W_t surges significantly
      compared to its historical average.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if input_path is None:
        input_path = os.path.join(base_dir, "data", "processed", "reddit_cleaned_6weeks.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "processed", "baseline_volume_alerts_6weeks.csv")

    print(f"1. Loading processed data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Total posts to analyze: {len(df)}")
    
    windows = sorted(df["window_id"].unique(), key=lambda w: int(w.split("_")[1]))
    print(f"   Total windows to analyze: {len(windows)}")

    # Window-by-window n-gram frequency counter
    window_term_counts = {}
    for win in windows:
        win_df = df[df["window_id"] == win]
        term_counter = Counter()
        for text in win_df["clean_text"]:
            for ngram in extract_ngrams(text):
                term_counter[ngram] += 1
        window_term_counts[win] = term_counter

    # Calculate baseline growth / surge for each term over time
    alerts = []
    all_terms = set()
    for counts in window_term_counts.values():
        all_terms.update(counts.keys())

    print("2. Detecting volume bursts across time windows...")
    for term in all_terms:
        history = []
        for t_idx, win in enumerate(windows):
            current_count = window_term_counts[win].get(term, 0)
            
            # If we have at least 1 previous window to establish historical volume
            if t_idx > 0 and current_count >= min_freq:
                avg_past_volume = sum(history) / len(history) if history else 0
                
                # Check if current volume surged significantly above past volume
                if avg_past_volume > 0:
                    surge_ratio = current_count / avg_past_volume
                else:
                    surge_ratio = float(current_count)  # Brand new appearance

                if surge_ratio >= surge_threshold:
                    alerts.append({
                        "term": term,
                        "detected_window": win,
                        "window_index": t_idx,
                        "current_volume": current_count,
                        "past_avg_volume": round(avg_past_volume, 2),
                        "surge_ratio": round(surge_ratio, 2)
                    })
                    break  # Record first time detected by volume baseline
            
            history.append(current_count)

    alert_df = pd.DataFrame(alerts)
    if not alert_df.empty:
        alert_df = alert_df.sort_values(by=["window_index", "surge_ratio"], ascending=[True, False]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    alert_df.to_csv(output_path, index=False)
    print(f"3. Volume Baseline generated {len(alert_df)} trend alerts.")
    print(f"   Saved baseline alerts to: {output_path}")
    return alert_df

if __name__ == "__main__":
    alerts = run_volume_baseline()
    if not alerts.empty:
        print("\n--- Sample Volume-Based Baseline Alerts ---")
        print(alerts.head(15).to_string())
    else:
        print("No alerts met the threshold in this slice.")
