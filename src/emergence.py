import os
import sys
import re
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def score_topic_trajectories(
    trajectories_path: str = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Step 6A Part 5.3: Research-Gated Emergence & Lifecycle Classification
    
    Implements 3 Strict Quality Guardrails:
    1. Temporal Gap <= 1 checkpoint (enforced upstream in temporal.py).
    2. Coherence Floor >= 0.62 (Rejects drifting/diffuse clusters like TRAJ_016).
       - Both avg_coherence >= 0.62 AND min_coherence >= 0.60.
    3. Measurable Single-Checkpoint Burst / Shockwave Category:
       - Defined by: Novel appearance (birth > W0) + Coherence >= 0.62 + Volume >= 15.
       - Captures high-impact events (e.g. OnlyFans 34 posts in W2).
       
    Lifecycle Categories:
    - [EMERGING] : Multi-week, net growth > 0, growth ratio >= 1.3, momentum >= 1.0, coherence >= 0.62
    - [GROWING]  : Multi-week, net growth > 0, steady growth, coherence >= 0.62
    - [BURST]    : Novel high-volume shockwave (vol >= 15, coherence >= 0.62, single window)
    - [DECLINING]: Shrinking, cooling, or stagnant volume
    - [REJECTED] : Low coherence / semantic drift (min_coherence < 0.60 or avg_coherence < 0.62)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if trajectories_path is None:
        trajectories_path = os.path.join(base_dir, "data", "processed", "topic_trajectories_6weeks.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "processed", "emerging_topics_alerts_6weeks.csv")

    print(f"1. Loading 1-to-1 tracked trajectories from: {trajectories_path}")
    df = pd.read_csv(trajectories_path)
    print(f"   Total trajectories to evaluate: {len(df)}")

    print("2. Applying Quality Guardrails & Lifecycle Classification...")
    scored_rows = []

    for _, r in df.iterrows():
        vol_hist = [int(x) for x in str(r["volume_path"]).split(" -> ")]
        n_ckpt = len(vol_hist)
        initial_vol = vol_hist[0]
        latest_vol = vol_hist[-1]
        peak_vol = max(vol_hist)
        total_vol = int(r["total_volume"])
        net_growth = latest_vol - initial_vol
        growth_ratio = latest_vol / (initial_vol + 1e-6)
        avg_coh = float(r["avg_coherence"])
        min_coh = float(r.get("min_coherence", avg_coh))
        birth_idx = int(r["birth_window_index"])

        # Recent acceleration (momentum between last 2 checkpoints)
        recent_momentum = (latest_vol / (vol_hist[-2] + 1e-6)) if n_ckpt >= 2 else 1.0

        # GUARDRAIL 2: Coherence Floor (avg >= 0.62 AND min >= 0.60)
        # Prevents semantic drift and catch-all clusters (e.g. TRAJ_016 at 0.546, TRAJ_004 at 0.548)
        if avg_coh < 0.62 or min_coh < 0.60:
            lifecycle_status = "[REJECTED (LOW_COHERENCE)]"
            growth_multiplier = 0.0
            raw_score = 0.0
            is_candidate = False

        # GUARDRAIL 3: Single-Checkpoint Burst / Shockwave
        elif n_ckpt == 1 and total_vol >= 15 and birth_idx > 0:
            lifecycle_status = "[BURST]"
            # High-volume sudden shockwave: score reflects size & coherence
            growth_multiplier = 1.35 + min(0.65, (total_vol - 15) / 30.0)
            coherence_score = max(0.0, min(1.0, (avg_coh - 0.50) / 0.35))
            size_conf = min(1.0, np.sqrt(total_vol) / 5.5)
            raw_score = growth_multiplier * coherence_score * size_conf
            is_candidate = True

        # Multi-checkpoint trajectories (n_ckpt >= 2) with Coherence >= 0.62
        elif n_ckpt >= 2:
            coherence_score = max(0.0, min(1.0, (avg_coh - 0.50) / 0.35))
            size_conf = min(1.0, np.sqrt(total_vol) / 6.0)
            persistence_score = min(1.0, n_ckpt / 3.0)
            birth_novelty = 1.0 if birth_idx > 0 else 0.70

            base_quality = (0.35 * persistence_score + 0.35 * coherence_score + 0.30 * birth_novelty)

            if net_growth > 0 and growth_ratio >= 1.3 and recent_momentum >= 1.0:
                lifecycle_status = "[EMERGING]"
                growth_multiplier = 1.10 + min(1.5, (growth_ratio - 1.0) / 1.1)
                raw_score = growth_multiplier * base_quality * size_conf
                is_candidate = True
            elif net_growth > 0:
                lifecycle_status = "[GROWING]"
                growth_multiplier = 0.90 + min(0.4, (growth_ratio - 1.0) / 2.0)
                raw_score = growth_multiplier * base_quality * size_conf
                is_candidate = True
            elif net_growth == 0:
                lifecycle_status = "[STAGNANT]"
                growth_multiplier = 0.50
                raw_score = growth_multiplier * base_quality * size_conf
                is_candidate = False
            else:
                lifecycle_status = "[DECLINING]"
                growth_multiplier = max(0.05, float(growth_ratio ** 1.8) * 0.40)
                raw_score = growth_multiplier * base_quality * size_conf
                is_candidate = False

        # Single-checkpoint below burst threshold (n_ckpt == 1, vol < 15 or birth_idx == 0)
        else:
            lifecycle_status = "[DECLINING]"
            growth_multiplier = 0.15
            coherence_score = max(0.0, min(1.0, (avg_coh - 0.50) / 0.35))
            raw_score = growth_multiplier * coherence_score * 0.3
            is_candidate = False

        normalized_emergence_score = min(1.0, raw_score / 1.25)

        # Clean topic label: filter out contraction artifacts (like 'didn')
        topic_name = str(r["topic_name"])
        topic_name = re.sub(r'\b(didn|doesn|wasn|aren|haven|hasn|wouldn|couldn)\b', '', topic_name)
        topic_name = re.sub(r'\s*/\s*/\s*', ' / ', topic_name).strip(' /')

        scored_rows.append({
            "trajectory_id": r["trajectory_id"],
            "topic_name": topic_name,
            "lifecycle_status": lifecycle_status,
            "is_vetted_candidate": is_candidate,
            "birth_window": r["birth_window"],
            "birth_window_index": birth_idx,
            "latest_window": r["latest_window"],
            "active_checkpoints": n_ckpt,
            "timeline_path": r["timeline_path"],
            "volume_path": r["volume_path"],
            "total_volume": total_vol,
            "net_volume_growth": net_growth,
            "growth_ratio": round(growth_ratio, 2),
            "avg_coherence": round(avg_coh, 3),
            "min_coherence": round(min_coh, 3),
            "emergence_score": round(float(normalized_emergence_score), 4),
            "growth_multiplier": round(float(growth_multiplier), 2),
            "representative_post": r["representative_post"]
        })

    scored_df = pd.DataFrame(scored_rows)
    scored_df = scored_df.sort_values(
        by=["is_vetted_candidate", "emergence_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    print(f"3. Saved vetted emergence alerts to: {output_path}")

    # Display Candidates
    candidates = scored_df[scored_df["is_vetted_candidate"] == True]
    print("\n" + "=" * 78)
    print(f"    RESEARCH-VERIFIED TOPIC CANDIDATES ({len(candidates)} VETTED TOPICS)")
    print("=" * 78)

    for rank, (_, row) in enumerate(candidates.iterrows(), 1):
        print(f"\n#{rank} [{row['trajectory_id']}] {row['lifecycle_status']} | Topic: '{row['topic_name']}'")
        print(f"    Emergence Score: {row['emergence_score']:.3f} | Avg Coh: {row['avg_coherence']:.3f} | Min Coh: {row['min_coherence']:.3f} | Total Vol: {row['total_volume']}")
        print(f"    Timeline Path:   {row['timeline_path']}")
        print(f"    Volume Path:     {row['volume_path']} (Net: {row['net_volume_growth']:+d}, Ratio: {row['growth_ratio']}x)")
        print(f"    Representative:  \"{row['representative_post']}\"")

    # Display Rejected / Non-candidate summary
    low_coh = scored_df[scored_df["lifecycle_status"] == "[REJECTED (LOW_COHERENCE)]"]
    declining = scored_df[scored_df["lifecycle_status"] == "[DECLINING]"]
    print("\n" + "-" * 78)
    print(f"Quality Filter Audit: {len(low_coh)} low-coherence topics rejected | {len(declining)} declining topics filtered")
    print("-" * 78)

    return scored_df

if __name__ == "__main__":
    score_topic_trajectories()
