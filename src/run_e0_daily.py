import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))

from clustering import extract_clean_ctf_idf, SPAM_REGEX, REDDIT_TROPES
from temporal import track_topic_trajectories
from emergence import score_topic_trajectories
from baseline import run_volume_baseline

def run_e0_clustering(embeddings_path, meta_path, clusters_out, summary_out):
    print("=" * 80)
    print("   STAGE 1: E0 TOPIC CLUSTERING ON 30-CHECKPOINT STREAM")
    print("=" * 80)
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(meta_path)
    df["cluster_id"] = -1
    df["cluster_name"] = "Noise"

    windows = sorted(df["window_id"].unique(), key=lambda w: int(w.split("_")[1]))
    print(f"Total checkpoints: {len(windows)} ({windows[0]} -> {windows[-1]})")

    cluster_summaries = []

    for win_idx, win in enumerate(windows):
        mask = (df["window_id"] == win).values
        win_emb = embeddings[mask]
        win_df = df[mask]
        week_label = win_df["week_label"].iloc[0] if "week_label" in win_df.columns else win

        if len(win_emb) < 7:
            continue

        # SVD 384D -> 12D
        svd = TruncatedSVD(n_components=12, random_state=42)
        reduced_emb = svd.fit_transform(win_emb)

        # HDBSCAN leaf selection
        clusterer = HDBSCAN(
            min_cluster_size=7,
            min_samples=3,
            cluster_selection_method='leaf',
            metric='euclidean',
            copy=True
        )
        labels = clusterer.fit_predict(reduced_emb)

        # Quality filtering & c-TF-IDF
        unique_labels = sorted([l for l in set(labels) if l != -1])
        valid_cluster_texts = {}
        for l in unique_labels:
            c_mask = (labels == l)
            c_texts = win_df.iloc[c_mask]["clean_text"].tolist()
            spam_count = sum(1 for t in c_texts if SPAM_REGEX.search(t))
            if (spam_count / len(c_texts)) > 0.35:
                labels[c_mask] = -1
                continue
            if len(c_texts) > 150:
                labels[c_mask] = -1
                continue
            valid_cluster_texts[l] = c_texts

        cluster_keywords = extract_clean_ctf_idf(valid_cluster_texts, top_k=3)

        for l in list(valid_cluster_texts.keys()):
            c_mask = (labels == l)
            c_indices = np.where(c_mask)[0]
            c_vectors = win_emb[c_indices]
            centroid = np.mean(c_vectors, axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            sims = cosine_similarity(c_vectors, centroid).flatten()
            avg_coherence = float(np.mean(sims))

            if avg_coherence < 0.45:
                labels[c_mask] = -1
                continue

            c_label = cluster_keywords.get(l, f"topic_{l}")
            rep_idx = c_indices[np.argmax(sims)]
            rep_post = win_df.iloc[rep_idx]["clean_text"]
            c_key = f"{win}_C{l}"

            cluster_summaries.append({
                "cluster_key": c_key,
                "window_index": win_idx,
                "window_id": win,
                "week_label": week_label,
                "cluster_id": l,
                "cluster_label": c_label,
                "post_count": int(np.sum(c_mask)),
                "avg_semantic_coherence": round(avg_coherence, 4),
                "representative_post": rep_post
            })

        df.loc[mask, "cluster_id"] = labels

    summary_df = pd.DataFrame(cluster_summaries)
    os.makedirs(os.path.dirname(clusters_out), exist_ok=True)
    df.to_csv(clusters_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    print(f"Clustering complete: Discovered {len(summary_df)} clusters across 30 checkpoints.")
    return summary_df

def run_e0_evaluation():
    e0_dir = os.path.join(base_dir, "experiments", "e0_control")
    os.makedirs(e0_dir, exist_ok=True)

    embeddings_path = os.path.join(base_dir, "data", "processed", "daily", "embeddings_daily.npy")
    meta_path = os.path.join(base_dir, "data", "processed", "daily", "embeddings_metadata_daily.csv")
    clusters_out = os.path.join(e0_dir, "daily_clusters.csv")
    summary_out = os.path.join(e0_dir, "daily_cluster_summary.csv")
    trajs_out = os.path.join(e0_dir, "daily_trajectories.csv")
    matches_out = os.path.join(e0_dir, "daily_trajectory_matches.csv")
    alerts_out = os.path.join(e0_dir, "emergence_alerts.csv")
    baseline_out = os.path.join(e0_dir, "baseline_volume_alerts_daily.csv")
    summary_txt = os.path.join(e0_dir, "e0_summary.txt")

    # 1. Clustering
    run_e0_clustering(embeddings_path, meta_path, clusters_out, summary_out)

    # 2. Temporal Tracking (Unmodified E0 logic with gap <= 1)
    print("\n" + "=" * 80)
    print("   STAGE 2: E0 1-TO-1 TEMPORAL TOPIC TRACKING")
    print("=" * 80)
    track_topic_trajectories(
        clusters_path=summary_out,
        meta_path=clusters_out,
        embeddings_path=embeddings_path,
        output_trajectories_path=trajs_out,
        output_matches_path=matches_out,
        match_threshold=0.68
    )

    # 3. Emergence & Burst Scoring (Unmodified E0 logic)
    print("\n" + "=" * 80)
    print("   STAGE 3: E0 EMERGENCE & BURST SCORING")
    print("=" * 80)
    score_topic_trajectories(
        trajectories_path=trajs_out,
        output_path=alerts_out
    )

    # 4. Volume Baseline
    print("\n" + "=" * 80)
    print("   STAGE 4: KEYWORD FREQUENCY BASELINE ON 30-CHECKPOINT STREAM")
    print("=" * 80)
    base_df = run_volume_baseline(
        input_path=meta_path,
        output_path=baseline_out,
        min_freq=4,
        surge_threshold=2.0
    )

    # 5. Targeted Event Analysis
    print("\n" + "=" * 80)
    print("   STAGE 5: TARGETED EVENT ANALYSIS (OnlyFans, FDA/Pfizer, Reddit Lock)")
    print("=" * 80)
    
    trajs_df = pd.read_csv(alerts_out)
    clusters_df = pd.read_csv(summary_out)

    events = [
        {
            "name": "OnlyFans Porn Ban",
            "date": "2021-08-19 (D_10/D_11)",
            "keywords": ["onlyfans", "porn", "content", "october", "explicit"],
            "pre_checkpoints": ["D_07", "D_08", "D_09"],
            "target_checkpoints": ["D_10", "D_11", "D_12"]
        },
        {
            "name": "FDA Pfizer Vaccine Approval",
            "date": "2021-08-23 (D_15)",
            "keywords": ["fda", "vaccine", "pfizer", "approved", "approval", "covid"],
            "pre_checkpoints": ["D_12", "D_13", "D_14"],
            "target_checkpoints": ["D_15", "D_16", "D_17"]
        },
        {
            "name": "Reddit Locked Comments Incident",
            "date": "2021-09-01/02 (D_25/D_26)",
            "keywords": ["lock", "locked", "locking", "comments", "vote", "mod"],
            "pre_checkpoints": ["D_22", "D_23", "D_24"],
            "target_checkpoints": ["D_25", "D_26", "D_27"]
        }
    ]

    report_lines = []
    report_lines.append("================================================================================")
    report_lines.append("        E0 CONTROL: 30-CHECKPOINT FINE-GRAINED TEMPORAL STREAM RESULTS          ")
    report_lines.append("================================================================================\n")

    for ev in events:
        ename = ev["name"]
        edate = ev["date"]
        kws = ev["keywords"]

        # 1. Baseline First Detection
        b_hits = base_df[base_df["term"].apply(lambda t: any(re.search(r'\b' + re.escape(kw) + r'\b', str(t).lower()) for kw in kws))]
        if not b_hits.empty:
            b_first = b_hits.sort_values(by="window_index").iloc[0]
            b_win = b_first["detected_window"]
            b_win_idx = int(b_first["window_index"])
            b_term = b_first["term"]
            b_ratio = b_first["surge_ratio"]
        else:
            b_win = "NEVER"
            b_win_idx = None
            b_term = "None"
            b_ratio = None

        # 2. Semantic Topic First Detection (in summary clusters)
        c_hits = clusters_df[clusters_df["cluster_label"].apply(lambda l: any(kw in str(l).lower() for kw in kws))]
        if not c_hits.empty:
            c_first = c_hits.sort_values(by="window_index").iloc[0]
            sem_cand_win = c_first["window_id"]
            sem_cand_idx = int(c_first["window_index"])
            sem_cand_label = c_first["cluster_label"]
            sem_cand_coh = c_first["avg_semantic_coherence"]
            sem_cand_rep = c_first["representative_post"]
        else:
            sem_cand_win = "NEVER"
            sem_cand_idx = None
            sem_cand_label = "None"
            sem_cand_coh = 0.0
            sem_cand_rep = "None"

        # 3. Validated Emergence Alerts (filtered candidates)
        v_hits = trajs_df[trajs_df["is_vetted_candidate"] == True]
        v_event_hits = v_hits[v_hits["topic_name"].apply(lambda l: any(kw in str(l).lower() for kw in kws))]
        if not v_event_hits.empty:
            v_first = v_event_hits.sort_values(by="birth_window_index").iloc[0]
            v_win = v_first["birth_window"]
            v_win_idx = int(v_first["birth_window_index"])
            v_status = v_first["lifecycle_status"]
        else:
            v_win = "NEVER"
            v_win_idx = None
            v_status = "None"

        # Check pre-event incubation
        pre_c = clusters_df[
            clusters_df["window_id"].isin(ev["pre_checkpoints"]) &
            clusters_df["cluster_label"].apply(lambda l: any(kw in str(l).lower() for kw in kws))
        ]
        has_pre_signal = not pre_c.empty

        lead_cand = (b_win_idx - sem_cand_idx) if (b_win_idx is not None and sem_cand_idx is not None) else "N/A"
        lead_val = (b_win_idx - v_win_idx) if (b_win_idx is not None and v_win_idx is not None) else "N/A"

        report_lines.append(f"Event: {ename} (Documented Date: {edate})")
        report_lines.append(f"  * First Semantic Candidate Cluster:  {sem_cand_win} ('{sem_cand_label}', Coh: {sem_cand_coh:.3f})")
        report_lines.append(f"  * First Validated Emergence Alert:   {v_win} ({v_status})")
        report_lines.append(f"  * First Keyword Baseline Surge:      {b_win} (Term: '{b_term}', Surge: {b_ratio}x)")
        report_lines.append(f"  * Candidate Lead Time:               {lead_cand} checkpoints")
        report_lines.append(f"  * Validated Alert Lead Time:         {lead_val} checkpoints")
        report_lines.append(f"  * Sub-threshold Pre-Event Signal:    {'YES' if has_pre_signal else 'NO (No cluster formed in pre-window)'}")
        if has_pre_signal:
            for _, pc in pre_c.iterrows():
                report_lines.append(f"      - Pre-Cluster [{pc['window_id']}]: '{pc['cluster_label']}' (Size: {pc['post_count']}, Coh: {pc['avg_semantic_coherence']:.3f})")
                report_lines.append(f"        Sample: \"{pc['representative_post'][:95]}\"")
        report_lines.append("-" * 75)

    report_lines.append("\nSUMMARY STATISTICS:")
    report_lines.append(f"  Total Clusters Discovered Across 30 Checkpoints: {len(clusters_df)}")
    report_lines.append(f"  Total Trajectories Tracked (1-to-1):             {len(trajs_df)}")
    report_lines.append(f"  Total Vetted Emergence Candidates:              {len(trajs_df[trajs_df['is_vetted_candidate'] == True])}")
    report_lines.append(f"  Total Keyword Baseline Alerts:                  {len(base_df)}")
    report_lines.append("================================================================================")

    full_report = "\n".join(report_lines)
    print(full_report)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\nSaved E0 Control summary report to: {summary_txt}")

if __name__ == "__main__":
    run_e0_evaluation()
