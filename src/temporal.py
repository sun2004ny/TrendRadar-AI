"""
src/temporal.py - Cross-temporal trajectory tracking and continuity gates
Uses pathlib, Python logging, and centralized config.
"""

from pathlib import Path
import re
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ROOT_DIR, get_config

logger = logging.getLogger(__name__)


def track_topic_trajectories(
    clusters_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    output_trajectories_path: Optional[Path] = None,
    output_matches_path: Optional[Path] = None,
    match_threshold: Optional[float] = None,
    max_gap_checkpoints: Optional[int] = None
) -> pd.DataFrame:
    """
    1-to-1 Cross-Checkpoint Topic Tracking & Cluster Trajectories:
    - STRICT 1-TO-1 MATCHING: Within any given window, a trajectory can match AT MOST one cluster.
    - LIFESPAN INTEGRITY: Clean volume paths without duplication.
    - CONTINUITY GATES: Cosine similarity >= threshold, gap constraint.
    """
    cfg = get_config()
    threshold = match_threshold if match_threshold is not None else float(cfg.temporal.get("match_threshold", 0.68))
    max_gap = max_gap_checkpoints if max_gap_checkpoints is not None else int(cfg.temporal.get("max_gap_checkpoints", 1))
    ema_alpha = float(cfg.temporal.get("centroid_ema_alpha", 0.65))

    if clusters_path is None:
        clusters_path = ROOT_DIR / "data" / "processed" / "clusters_summary_6weeks.csv"
    else:
        clusters_path = Path(clusters_path)

    if meta_path is None:
        meta_path = ROOT_DIR / "data" / "processed" / "clustered_topics_6weeks.csv"
    else:
        meta_path = Path(meta_path)

    if embeddings_path is None:
        embeddings_path = ROOT_DIR / "data" / "processed" / "embeddings_6weeks.npy"
    else:
        embeddings_path = Path(embeddings_path)

    if output_trajectories_path is None:
        output_trajectories_path = ROOT_DIR / "data" / "processed" / "topic_trajectories_6weeks.csv"
    else:
        output_trajectories_path = Path(output_trajectories_path)

    if output_matches_path is None:
        output_matches_path = ROOT_DIR / "data" / "processed" / "trajectory_matches_6weeks.csv"
    else:
        output_matches_path = Path(output_matches_path)

    logger.info("Loading refined clusters from: %s", clusters_path)
    cluster_df = pd.read_csv(clusters_path)
    posts_df = pd.read_csv(meta_path)
    embeddings = np.load(embeddings_path)

    # Precompute centroids
    centroids = {}
    for _, row in cluster_df.iterrows():
        c_key = row["cluster_key"]
        win_id = row["window_id"]
        c_num = int(c_key.split("_C")[1])
        mask = (posts_df["window_id"] == win_id) & (posts_df["cluster_id"] == c_num)
        c_emb = embeddings[mask.values]
        if len(c_emb) > 0:
            c_vec = np.mean(c_emb, axis=0, keepdims=True)
            c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-12)
            centroids[c_key] = c_vec

    windows = sorted(cluster_df["window_id"].unique(), key=lambda w: int(w.split("_")[1]))
    logger.info("Tracking trajectories across %d checkpoints (match threshold >= %.2f)...", len(windows), threshold)

    next_traj_id = 1
    trajectories = {}
    cluster_matches = []

    for win_idx, win in enumerate(windows):
        win_clusters = cluster_df[cluster_df["window_id"] == win]
        week_label = win_clusters["week_label"].iloc[0] if "week_label" in win_clusters.columns else win
        active_traj_ids = list(trajectories.keys())

        if win_idx == 0 or len(active_traj_ids) == 0:
            for _, c_row in win_clusters.iterrows():
                c_key = c_row["cluster_key"]
                c_vec = centroids[c_key]
                t_id = f"TRAJ_{next_traj_id:03d}"
                next_traj_id += 1

                trajectories[t_id] = {
                    "trajectory_id": t_id,
                    "canonical_name": c_row["cluster_label"],
                    "birth_window": win,
                    "birth_window_index": win_idx,
                    "latest_window": win,
                    "latest_window_index": win_idx,
                    "checkpoint_history": [win],
                    "volume_history": [int(c_row["post_count"])],
                    "running_centroid": c_vec,
                    "member_clusters": [c_key],
                    "representative_post": c_row["representative_post"],
                    "avg_coherence": float(c_row["avg_semantic_coherence"]),
                    "min_coherence": float(c_row["avg_semantic_coherence"])
                }
                cluster_matches.append({
                    "cluster_key": c_key,
                    "window_id": win,
                    "trajectory_id": t_id,
                    "match_similarity": 1.0,
                    "is_birth": True
                })
        else:
            # 1-to-1 bipartite matching for current window
            # Temporal gap constraint: win_idx - latest_win_idx <= max_gap + 1
            eligible_traj_ids = [
                tid for tid in active_traj_ids
                if (win_idx - trajectories[tid]["latest_window_index"]) <= (max_gap + 1)
            ]

            candidate_pairs = []
            if len(eligible_traj_ids) > 0:
                eligible_centroids = np.vstack([trajectories[tid]["running_centroid"] for tid in eligible_traj_ids])
                for c_idx, (_, c_row) in enumerate(win_clusters.iterrows()):
                    c_key = c_row["cluster_key"]
                    c_vec = centroids[c_key]
                    sims = cosine_similarity(c_vec, eligible_centroids).flatten()
                    for tid_idx, sim in enumerate(sims):
                        if sim >= threshold:
                            candidate_pairs.append({
                                "sim": float(sim),
                                "cluster_key": c_key,
                                "c_row": c_row,
                                "c_vec": c_vec,
                                "traj_id": eligible_traj_ids[tid_idx]
                            })

            candidate_pairs.sort(key=lambda x: x["sim"], reverse=True)
            assigned_clusters = set()
            claimed_trajectories_this_window = set()

            for pair in candidate_pairs:
                c_key = pair["cluster_key"]
                t_id = pair["traj_id"]

                if c_key not in assigned_clusters and t_id not in claimed_trajectories_this_window:
                    assigned_clusters.add(c_key)
                    claimed_trajectories_this_window.add(t_id)

                    t = trajectories[t_id]
                    t["latest_window"] = win
                    t["latest_window_index"] = win_idx
                    t["checkpoint_history"].append(win)
                    t["volume_history"].append(int(pair["c_row"]["post_count"]))
                    t["member_clusters"].append(c_key)

                    # Update running centroid EMA
                    new_vec = ema_alpha * t["running_centroid"] + (1.0 - ema_alpha) * pair["c_vec"]
                    t["running_centroid"] = new_vec / (np.linalg.norm(new_vec) + 1e-12)
                    c_coherence = float(pair["c_row"]["avg_semantic_coherence"])
                    t.setdefault("coherence_list", [t["avg_coherence"]]).append(c_coherence)
                    t["avg_coherence"] = float(np.mean(t["coherence_list"]))
                    t["min_coherence"] = float(min(t["coherence_list"]))

                    cluster_matches.append({
                        "cluster_key": c_key,
                        "window_id": win,
                        "trajectory_id": t_id,
                        "match_similarity": round(pair["sim"], 4),
                        "is_birth": False
                    })

            # Unassigned clusters form new trajectories
            for _, c_row in win_clusters.iterrows():
                c_key = c_row["cluster_key"]
                if c_key not in assigned_clusters:
                    c_vec = centroids[c_key]
                    t_id = f"TRAJ_{next_traj_id:03d}"
                    next_traj_id += 1

                    trajectories[t_id] = {
                        "trajectory_id": t_id,
                        "canonical_name": c_row["cluster_label"],
                        "birth_window": win,
                        "birth_window_index": win_idx,
                        "latest_window": win,
                        "latest_window_index": win_idx,
                        "checkpoint_history": [win],
                        "volume_history": [int(c_row["post_count"])],
                        "running_centroid": c_vec,
                        "member_clusters": [c_key],
                        "representative_post": c_row["representative_post"],
                        "avg_coherence": float(c_row["avg_semantic_coherence"]),
                        "min_coherence": float(c_row["avg_semantic_coherence"])
                    }
                    cluster_matches.append({
                        "cluster_key": c_key,
                        "window_id": win,
                        "trajectory_id": t_id,
                        "match_similarity": 1.0,
                        "is_birth": True
                    })

    # Prepare DataFrame
    traj_rows = []
    for t_id, t in trajectories.items():
        traj_rows.append({
            "trajectory_id": t["trajectory_id"],
            "topic_name": t["canonical_name"],
            "birth_window": t["birth_window"],
            "birth_window_index": t["birth_window_index"],
            "latest_window": t["latest_window"],
            "active_checkpoints": len(t["checkpoint_history"]),
            "timeline_path": " -> ".join(t["checkpoint_history"]),
            "volume_path": " -> ".join(str(v) for v in t["volume_history"]),
            "total_volume": sum(t["volume_history"]),
            "avg_coherence": round(t["avg_coherence"], 4),
            "min_coherence": round(t.get("min_coherence", t["avg_coherence"]), 4),
            "representative_post": t["representative_post"]
        })

    trajectories_df = pd.DataFrame(traj_rows)
    matches_df = pd.DataFrame(cluster_matches)

    output_trajectories_path.parent.mkdir(parents=True, exist_ok=True)
    output_matches_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories_df.to_csv(output_trajectories_path, index=False)
    matches_df.to_csv(output_matches_path, index=False)

    logger.info("Saved %d tracked trajectories to: %s", len(trajectories_df), output_trajectories_path)
    logger.info("Saved %d cluster matches to: %s", len(matches_df), output_matches_path)

    return trajectories_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    track_topic_trajectories()
