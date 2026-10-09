"""
src/novelty.py - Causal Background Chatter & Novelty Gate for TrendRadar-AI

Filters or demotes recurring AskReddit background chatter topics by checking
cosine similarity between candidate topic centroids and topics from earlier days.

Rules:
1. Strict Causality: For any candidate on day D_t, compare ONLY against centroids from D_tau (tau < t).
   Zero future data leakage.
2. Flag Off: When enabled=False, returns exact inputs unmodified (golden compatibility).
3. Tuned on D01-D20: Default threshold 0.70 chosen because on D01-D20:
   - Max similarity of real events to past topics was 0.598.
   - Noise topic similarities reached 0.924 (median 0.68).
   - At 0.70, zero real events are filtered while 40% of noise candidates on D01-D20 are quarantined.
"""

from pathlib import Path
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_config
from src.evaluation import parse_day_num

logger = logging.getLogger(__name__)


class NoveltyGate:
    """
    Causal novelty gate that identifies recurring background chatter.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        similarity_threshold: Optional[float] = None,
        action: Optional[str] = None,
        demote_penalty: Optional[float] = None,
        config: Optional[Any] = None
    ):
        cfg = config or get_config()
        ng_cfg = cfg.get("novelty_gate", {}) if hasattr(cfg, "get") else getattr(cfg, "novelty_gate", {})

        self.enabled = enabled if enabled is not None else bool(ng_cfg.get("enabled", False))
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else float(ng_cfg.get("similarity_threshold", 0.70))
        )
        self.action = (
            action if action is not None
            else str(ng_cfg.get("action", "quarantine"))
        )
        self.demote_penalty = (
            demote_penalty if demote_penalty is not None
            else float(ng_cfg.get("demote_penalty", 0.20))
        )

    def apply(
        self,
        alerts_df: pd.DataFrame,
        clusters_df: Optional[pd.DataFrame] = None,
        embeddings: Optional[np.ndarray] = None,
        matches_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies causal novelty evaluation to an alerts DataFrame.
        - If self.enabled is False: Returns alerts_df unmodified.
        - If self.enabled is True: Computes max cosine similarity to prior checkpoints (tau < t).
        """
        if not self.enabled:
            logger.info("NoveltyGate DISABLED: Returning alerts unmodified.")
            metrics = {
                "novelty_gate_enabled": False,
                "chatter_quarantined": 0,
                "chatter_demoted": 0
            }
            return alerts_df.copy(), metrics

        logger.info(
            "NoveltyGate ENABLED: Causal similarity check (threshold=%.2f, action=%s)...",
            self.similarity_threshold, self.action
        )

        df = alerts_df.copy()

        # Load required cluster data if not provided
        if clusters_df is None:
            c_path = ROOT_DIR / "experiments" / "e6_combined" / "daily_clusters.csv"
            clusters_df = pd.read_csv(c_path) if c_path.exists() else pd.DataFrame()

        if embeddings is None:
            e_path = ROOT_DIR / "data" / "processed" / "daily" / "embeddings_daily.npy"
            embeddings = np.load(e_path) if e_path.exists() else np.array([])

        if matches_df is None:
            m_path = ROOT_DIR / "experiments" / "e6_combined" / "daily_trajectory_matches.csv"
            matches_df = pd.read_csv(m_path) if m_path.exists() else pd.DataFrame()

        # Precompute centroids for each daily cluster (checkpoint id, cluster id)
        cluster_centroids: Dict[str, Tuple[str, int, np.ndarray]] = {}
        if len(clusters_df) > 0 and len(embeddings) > 0:
            for (win, cid), group in clusters_df.groupby(["window_id", "cluster_id"]):
                c_key = f"{win}_C{cid}"
                indices = group.index.values
                if len(indices) > 0 and max(indices) < len(embeddings):
                    c_embs = embeddings[indices]
                    cent = np.mean(c_embs, axis=0)
                    cent = cent / (np.linalg.norm(cent) + 1e-12)
                    cluster_centroids[c_key] = (win, parse_day_num(win), cent)

        # Map each trajectory_id to its initial birth centroid
        traj_centroids: Dict[str, Tuple[str, int, np.ndarray]] = {}
        if len(matches_df) > 0:
            for tid, group in matches_df.groupby("trajectory_id"):
                birth_matches = group[group["is_birth"] == True]
                if len(birth_matches) == 0:
                    birth_matches = group.head(1)
                b_key = birth_matches.iloc[0]["cluster_key"]
                if b_key in cluster_centroids:
                    traj_centroids[str(tid)] = cluster_centroids[b_key]

        chatter_quarantined = 0
        chatter_demoted = 0

        # Causal loop: For each candidate alert born on day t,
        # compare only against cluster centroids from days strictly < t
        for idx, row in df.iterrows():
            tid = str(row.get("trajectory_id", ""))
            day_str = str(row.get("birth_window", row.get("detected_window", "D_01")))
            t_day_num = parse_day_num(day_str)

            # Retrieve or compute centroid
            cent = None
            if tid in traj_centroids:
                _, _, cent = traj_centroids[tid]
            elif "representative_post" in row and len(clusters_df) > 0 and len(embeddings) > 0:
                rep_p = str(row["representative_post"])
                p_match = clusters_df[clusters_df["clean_text"] == rep_p]
                if len(p_match) > 0:
                    p_idx = p_match.index.values[0]
                    if p_idx < len(embeddings):
                        cent = embeddings[p_idx] / (np.linalg.norm(embeddings[p_idx]) + 1e-12)

            if cent is not None:
                # Causal check: ONLY prior days (d < t_day_num)
                prior_cents = [c for (w, d, c) in cluster_centroids.values() if d < t_day_num]
                if len(prior_cents) > 0:
                    sims = np.dot(prior_cents, cent)
                    max_sim = float(np.max(sims))
                else:
                    max_sim = 0.0
            else:
                max_sim = 0.0

            df.at[idx, "max_prior_sim"] = round(max_sim, 4)
            df.at[idx, "novelty_score"] = round(float(max(0.0, 1.0 - max_sim)), 4)

            is_chatter = max_sim >= self.similarity_threshold
            df.at[idx, "is_background_chatter"] = is_chatter

            curr_score = float(row.get("emergence_score", 0.0))

            if is_chatter:
                if self.action == "quarantine":
                    df.at[idx, "is_vetted_candidate"] = False
                    df.at[idx, "lifecycle_status"] = "[RECURRING_CHATTER]"
                    df.at[idx, "emergence_ranking_score"] = round(curr_score * self.demote_penalty, 4)
                    chatter_quarantined += 1
                else:  # demote
                    df.at[idx, "emergence_ranking_score"] = round(curr_score * self.demote_penalty, 4)
                    chatter_demoted += 1
            else:
                # Genuinely novel topics retain full emergence score
                df.at[idx, "emergence_ranking_score"] = round(curr_score, 4)

        # Sort so that top 10 alerts prioritize non-chatter novel events
        sort_col = "emergence_ranking_score" if "emergence_ranking_score" in df.columns else "emergence_score"
        df = df.sort_values(
            by=["is_vetted_candidate", sort_col],
            ascending=[False, False]
        ).reset_index(drop=True)

        metrics = {
            "novelty_gate_enabled": True,
            "similarity_threshold": self.similarity_threshold,
            "action": self.action,
            "chatter_quarantined": chatter_quarantined,
            "chatter_demoted": chatter_demoted,
            "remaining_candidates": int((df["is_vetted_candidate"] == True).sum())
        }

        return df, metrics


def apply_novelty_gate(
    alerts_df: pd.DataFrame,
    enabled: Optional[bool] = None,
    similarity_threshold: Optional[float] = None,
    action: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Helper functional interface to apply NoveltyGate."""
    gate = NoveltyGate(
        enabled=enabled,
        similarity_threshold=similarity_threshold,
        action=action
    )
    return gate.apply(alerts_df)
