import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def track_topic_trajectories(
    clusters_path: str = None,
    meta_path: str = None,
    embeddings_path: str = None,
    output_trajectories_path: str = None,
    output_matches_path: str = None,
    match_threshold: float = 0.68
):
    """
    Step 6A Part 5.5: 1-to-1 Cross-Week Topic Tracking & Cluster Trajectories
    
    Guarantees:
    1. STRICT 1-TO-1 MATCHING: Within any given week, a trajectory can match AT MOST
       one cluster. Multiple clusters from the same week can never attach to the same trajectory.
    2. LIFESPAN INTEGRITY: Active Lifespan is the number of distinct weekly checkpoints
       (maximum 6 weeks).
    3. CLEAN VOLUME PATH: Exactly one volume measurement per active checkpoint.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if clusters_path is None:
        clusters_path = os.path.join(base_dir, "data", "processed", "clusters_summary_6weeks.csv")
    if meta_path is None:
        meta_path = os.path.join(base_dir, "data", "processed", "clustered_topics_6weeks.csv")
    if embeddings_path is None:
        embeddings_path = os.path.join(base_dir, "data", "processed", "embeddings_6weeks.npy")
    if output_trajectories_path is None:
        output_trajectories_path = os.path.join(base_dir, "data", "processed", "topic_trajectories_6weeks.csv")
    if output_matches_path is None:
        output_matches_path = os.path.join(base_dir, "data", "processed", "trajectory_matches_6weeks.csv")

    print(f"1. Loading refined clusters from: {clusters_path}")
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
    print(f"2. Tracking trajectories across {len(windows)} weekly checkpoints (1-to-1 matching, threshold >= {match_threshold})...")

    next_traj_id = 1
    trajectories = {}  # traj_id -> dict of trajectory state
    cluster_matches = []  # cluster_key -> traj_id, sim_score

    for win_idx, win in enumerate(windows):
        win_clusters = cluster_df[cluster_df["window_id"] == win]
        week_label = win_clusters["week_label"].iloc[0] if "week_label" in win_clusters.columns else win
        
        active_traj_ids = list(trajectories.keys())

        # If first window (W0), all clusters start fresh trajectories
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
                    "avg_coherence": float(c_row["avg_semantic_coherence"])
                }
                cluster_matches.append({
                    "cluster_key": c_key,
                    "window_id": win,
                    "trajectory_id": t_id,
                    "match_similarity": 1.0,
                    "is_birth": True
                })
        else:
            # 1-TO-1 GREEDY BIPARTITE MATCHING FOR CURRENT WINDOW
            # Compute pairwise similarities between all clusters in current window
            # and all active existing trajectories
            # GUARDRAIL 1: Temporal gap <= 1 checkpoint (curr_win_idx - latest_win_idx <= 2)
            # Rejects long-gap accidental semantic matches (e.g. W0 -> W4)
            eligible_traj_ids = [
                tid for tid in active_traj_ids
                if (win_idx - trajectories[tid]["latest_window_index"]) <= 2
            ]

            candidate_pairs = []
            if len(eligible_traj_ids) > 0:
                eligible_centroids = np.vstack([trajectories[tid]["running_centroid"] for tid in eligible_traj_ids])
                for c_idx, (_, c_row) in enumerate(win_clusters.iterrows()):
                    c_key = c_row["cluster_key"]
                    c_vec = centroids[c_key]
                    sims = cosine_similarity(c_vec, eligible_centroids).flatten()
                    for tid_idx, sim in enumerate(sims):
                        if sim >= match_threshold:
                            candidate_pairs.append({
                                "sim": float(sim),
                                "cluster_key": c_key,
                                "c_row": c_row,
                                "c_vec": c_vec,
                                "traj_id": eligible_traj_ids[tid_idx]
                            })

            # Sort candidate pairs by highest similarity first
            candidate_pairs.sort(key=lambda x: x["sim"], reverse=True)

            assigned_clusters = set()
            claimed_trajectories_this_window = set()

            for pair in candidate_pairs:
                c_key = pair["cluster_key"]
                t_id = pair["traj_id"]

                # Strict 1-to-1 constraint:
                # Neither the cluster nor the trajectory can be matched twice in this window
                if c_key not in assigned_clusters and t_id not in claimed_trajectories_this_window:
                    assigned_clusters.add(c_key)
                    claimed_trajectories_this_window.add(t_id)

                    t = trajectories[t_id]
                    t["latest_window"] = win
                    t["latest_window_index"] = win_idx
                    t["checkpoint_history"].append(win)
                    t["volume_history"].append(int(pair["c_row"]["post_count"]))
                    t["member_clusters"].append(c_key)
                    
                    # Update running centroid
                    new_vec = 0.65 * t["running_centroid"] + 0.35 * pair["c_vec"]
                    t["running_centroid"] = new_vec / (np.linalg.norm(new_vec) + 1e-12)
                    c_coherence = float(pair["c_row"]["avg_semantic_coherence"])
                    t.setdefault("coherence_list", [t["avg_coherence"]]).append(c_coherence)
                    t["avg_coherence"] = float(np.mean(t["coherence_list"]))

                    cluster_matches.append({
                        "cluster_key": c_key,
                        "window_id": win,
                        "trajectory_id": t_id,
                        "match_similarity": round(pair["sim"], 4),
                        "is_birth": False
                    })

            # Any unassigned clusters in this window become the BIRTH of a new trajectory!
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
                        "avg_coherence": float(c_row["avg_semantic_coherence"])
                    }
                    cluster_matches.append({
                        "cluster_key": c_key,
                        "window_id": win,
                        "trajectory_id": t_id,
                        "match_similarity": 0.0,
                        "is_birth": True
                    })

    # Prepare Trajectories Summary DataFrame
    traj_rows = []
    for t_id, t in trajectories.items():
        distinct_checkpoints = len(t["checkpoint_history"])  # Strictly <= 6
        initial_vol = t["volume_history"][0]
        peak_vol = max(t["volume_history"])
        total_vol = sum(t["volume_history"])
        latest_vol = t["volume_history"][-1]
        
        vol_growth = (latest_vol - initial_vol) if distinct_checkpoints > 1 else 0
        growth_ratio = (latest_vol / (initial_vol + 1e-6)) if distinct_checkpoints > 1 else 1.0

        coherences = t.get("coherence_list", [t["avg_coherence"]])
        min_coh = float(min(coherences))
        avg_coh = float(np.mean(coherences))

        traj_rows.append({
            "trajectory_id": t_id,
            "topic_name": t["canonical_name"],
            "birth_window": t["birth_window"],
            "birth_window_index": t["birth_window_index"],
            "latest_window": t["latest_window"],
            "active_checkpoints": distinct_checkpoints,
            "timeline_path": " -> ".join(t["checkpoint_history"]),
            "volume_path": " -> ".join(map(str, t["volume_history"])),
            "initial_volume": initial_vol,
            "peak_volume": peak_vol,
            "latest_volume": latest_vol,
            "total_volume": total_vol,
            "net_volume_growth": vol_growth,
            "growth_ratio": round(growth_ratio, 2),
            "avg_coherence": round(avg_coh, 3),
            "min_coherence": round(min_coh, 3),
            "representative_post": t["representative_post"]
        })

    traj_df = pd.DataFrame(traj_rows)
    matches_df = pd.DataFrame(cluster_matches)

    os.makedirs(os.path.dirname(output_trajectories_path), exist_ok=True)
    traj_df.to_csv(output_trajectories_path, index=False)
    matches_df.to_csv(output_matches_path, index=False)

    print(f"\n3. Saved 1-to-1 tracked trajectories to: {output_trajectories_path}")
    print(f"   Saved cluster matching log to: {output_matches_path}")

    # Summary
    persisting_trajs = traj_df[traj_df["active_checkpoints"] > 1]
    single_window_trajs = traj_df[traj_df["active_checkpoints"] == 1]
    print("\n--- Verified Trajectory Analysis (Max 6 Checkpoints) ---")
    print(f"Total Evolving Trajectories Discovered: {len(traj_df)}")
    print(f"Multi-Week Persisting Topics (>= 2 weeks): {len(persisting_trajs)}")
    print(f"Single-Checkpoint Isolated Topics:       {len(single_window_trajs)}")

    if not persisting_trajs.empty:
        print("\n--- Top Multi-Week Persistent & Growing Topics (Clean 1-to-1) ---")
        persisting_preview = persisting_trajs.sort_values(by=["active_checkpoints", "net_volume_growth"], ascending=[False, False]).head(10)
        for _, r in persisting_preview.iterrows():
            print(f"\n[{r['trajectory_id']}] '{r['topic_name']}'")
            print(f"   Birth: {r['birth_window']} | Active Lifespan: {r['active_checkpoints']} distinct weeks (Max 6)")
            print(f"   Timeline Path: {r['timeline_path']}")
            print(f"   Volume Path:   {r['volume_path']} (Net Growth: {r['net_volume_growth']:+d}, Growth Ratio: {r['growth_ratio']}x)")
            print(f"   Sample: \"{r['representative_post']}\"")

    return traj_df, matches_df

if __name__ == "__main__":
    track_topic_trajectories()
