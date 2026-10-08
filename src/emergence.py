"""
src/emergence.py - Emergence scoring, quality guardrails, and lifecycle classification
Uses pathlib, Python logging, and centralized config.
"""

from pathlib import Path
import re
import logging
from typing import Optional
import pandas as pd
import numpy as np

from src.config import ROOT_DIR, get_config

logger = logging.getLogger(__name__)


def score_topic_trajectories(
    trajectories_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Research-Gated Emergence & Lifecycle Classification:
    1. Temporal Gap <= 1 checkpoint (enforced upstream in temporal.py).
    2. Coherence Floor >= 0.62 (avg >= 0.62 AND min >= 0.60).
    3. Measurable Single-Checkpoint Burst / Shockwave (Volume >= 15, Coherence >= 0.62).
    """
    cfg = get_config()
    coh_floor = float(cfg.emergence.get("coherence_floor", 0.62))
    min_coh_floor = float(cfg.emergence.get("min_coherence_floor", 0.60))
    burst_vol = int(cfg.emergence.get("burst_static_volume", 15))
    growth_ratio_min = float(cfg.emergence.get("growth_ratio_threshold", 1.3))

    if trajectories_path is None:
        trajectories_path = ROOT_DIR / "data" / "processed" / "topic_trajectories_6weeks.csv"
    else:
        trajectories_path = Path(trajectories_path)

    if output_path is None:
        output_path = ROOT_DIR / "data" / "processed" / "emerging_topics_alerts_6weeks.csv"
    else:
        output_path = Path(output_path)

    logger.info("Loading trajectories from: %s", trajectories_path)
    df = pd.read_csv(trajectories_path)
    logger.info("Total trajectories to evaluate: %d", len(df))

    scored_rows = []

    for _, r in df.iterrows():
        vol_hist = [int(x) for x in str(r["volume_path"]).split(" -> ")]
        n_ckpt = len(vol_hist)
        initial_vol = vol_hist[0]
        latest_vol = vol_hist[-1]
        total_vol = int(r["total_volume"])
        net_growth = latest_vol - initial_vol
        growth_ratio = latest_vol / (initial_vol + 1e-6)
        avg_coh = float(r["avg_coherence"])
        min_coh = float(r.get("min_coherence", avg_coh))
        birth_idx = int(r["birth_window_index"])

        recent_momentum = (latest_vol / (vol_hist[-2] + 1e-6)) if n_ckpt >= 2 else 1.0

        # GUARDRAIL 2: Coherence Floor (avg >= 0.62 AND min >= 0.60)
        if avg_coh < coh_floor or min_coh < min_coh_floor:
            lifecycle_status = "[REJECTED (LOW_COHERENCE)]"
            growth_multiplier = 0.0
            raw_score = 0.0
            is_candidate = False

        # GUARDRAIL 3: Single-Checkpoint Burst / Shockwave
        elif n_ckpt == 1 and total_vol >= burst_vol and birth_idx > 0:
            lifecycle_status = "[BURST]"
            growth_multiplier = 1.35 + min(0.65, (total_vol - burst_vol) / 30.0)
            coherence_score = max(0.0, min(1.0, (avg_coh - 0.50) / 0.35))
            size_conf = min(1.0, np.sqrt(total_vol) / 5.5)
            raw_score = growth_multiplier * coherence_score * size_conf
            is_candidate = True

        # Multi-checkpoint trajectories (n_ckpt >= 2) with Coherence >= floor
        elif n_ckpt >= 2:
            coherence_score = max(0.0, min(1.0, (avg_coh - 0.50) / 0.35))
            size_conf = min(1.0, np.sqrt(total_vol) / 6.0)
            persistence_score = min(1.0, n_ckpt / 3.0)
            birth_novelty = 1.0 if birth_idx > 0 else 0.70

            base_quality = (0.35 * persistence_score + 0.35 * coherence_score + 0.30 * birth_novelty)

            if net_growth > 0 and growth_ratio >= growth_ratio_min and recent_momentum >= 1.0:
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

        # Single-checkpoint below burst threshold
        else:
            lifecycle_status = "[DECLINING]"
            growth_multiplier = 0.15
            coherence_score = max(0.0, min(1.0, (avg_coh - 0.50) / 0.35))
            raw_score = growth_multiplier * coherence_score * 0.3
            is_candidate = False

        normalized_emergence_score = min(1.0, raw_score / 1.25)

        # Clean topic label: filter out contraction artifacts
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    logger.info("Saved %d scored emergence alerts to: %s", len(scored_df), output_path)

    return scored_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    score_topic_trajectories()
