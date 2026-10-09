#!/usr/bin/env python3
"""
scripts/reproduce_all.py
========================
Master End-to-End Reproduction & Benchmark Verification Script for TrendRadar-AI.

Executes:
1. Golden Baseline Integrity Verification (895 trajectories, 102 vetted baseline alerts, 135 buffer-vetted alerts)
2. Real Causal Day-by-Day Streaming Shadow Replay across D_01..D_30 (scripts/run_shadow_replay.py)
3. Full Out-of-Sample Empirical Metrics Calculation on Held-Out Days D_21..D_30 (Recall Denominator = 5 Ground Truth Events)
4. Comprehensive Automated Test Suite Execution (58 unit & regression tests)
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Set, Any, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_config
from src.rolling_buffer import run_pipeline_with_buffer
from src.evaluation import parse_day_num
from scripts.run_shadow_replay import run_real_shadow_replay


def step1_verify_pipeline_integrity():
    print("\n" + "=" * 75)
    print(" STEP 1: VERIFYING PIPELINE INTEGRITY & GOLDEN TRAJECTORIES")
    print("=" * 75)

    df_off, _ = run_pipeline_with_buffer(enabled=False)
    df_on, _ = run_pipeline_with_buffer(enabled=True)

    total_off = len(df_off)
    vetted_off = len(df_off[df_off["is_vetted_candidate"] == True])

    total_on = len(df_on)
    vetted_on = len(df_on[df_on["is_vetted_candidate"] == True])

    print(f"[*] Buffer OFF (Phase 2 Golden) : {total_off} total trajectories, {vetted_off} vetted candidates")
    print(f"[*] Buffer ON  (Phase 3 Golden) : {total_on} total trajectories, {vetted_on} vetted candidates (102 base + 33 seeds)")

    assert total_off == 862, f"Expected 862 total trajectories (buffer off), got {total_off}"
    assert vetted_off == 102, f"Expected 102 baseline vetted candidates, got {vetted_off}"
    assert total_on == 895, f"Expected 895 total trajectories (buffer on), got {total_on}"
    assert vetted_on == 135, f"Expected 135 buffer-vetted candidates, got {vetted_on}"

    print("[+] Step 1 PASSED: Pipeline output 895/135 preserved exactly.")


def step2_run_shadow_replay():
    print("\n" + "=" * 75)
    print(" STEP 2: RUNNING REAL CAUSAL DAY-BY-DAY SHADOW REPLAY")
    print("=" * 75)
    run_real_shadow_replay()
    print("[+] Step 2 PASSED: Real shadow replay finished and logged to logs/shadow_log.jsonl.")


def compute_metrics_from_sources(ground_truth_path: Path, labels_path: Path, baseline_path: Path, vetted_df: Optional[pd.DataFrame] = None):
    """
    Dynamically computes empirical evaluation metrics on D21-D30 from source files:
    - Ground Truth events (denominator) from ground_truth_path
    - Expert / LLM labels from labels_path
    - Keyword frequency alerts from baseline_path
    - Topic trajectories and novelty gate status from pipeline and vector centroids
    - Novelty threshold and temporal matching window loaded directly from config.yaml
    """
    from src.config import load_raw_config
    cfg = load_raw_config()
    lead_prior = int(cfg.get("evaluation", {}).get("lead_window_days_prior", 2))
    lag_post = int(cfg.get("evaluation", {}).get("lag_window_days_post", 3))
    sim_threshold = float(cfg.get("novelty_gate", {}).get("similarity_threshold", 0.70))

    # 1. Ground Truth Denominator on D21-D30
    gt_df = pd.read_csv(ground_truth_path)
    gt_d21_30 = gt_df[
        (gt_df["label"] == "real_event") &
        (gt_df["start_day"].apply(lambda d: parse_day_num(d) >= 21))
    ].copy()
    
    gt_event_ids = set(gt_d21_30["event_id"])
    total_gt_events = len(gt_event_ids)
    if total_gt_events == 0:
        raise ValueError("Ground truth file contains 0 real events on D21-D30.")

    gt_keywords = {}
    for _, er in gt_d21_30.iterrows():
        gt_keywords[er["event_id"]] = [k.strip().lower() for k in str(er["keywords"]).split(",") if k.strip()]

    # 2. Alert Label Map from review file
    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["alert_id"].astype(str), labels_df["my_label"].astype(str)))

    # 3. Keyword Volume Baseline on D21-D30 with temporal window matching from config
    base_alerts_df = pd.read_csv(baseline_path)
    base_d21_30 = base_alerts_df[base_alerts_df["detected_window"].apply(lambda w: parse_day_num(w) >= 21)].copy()
    base_count = len(base_d21_30)

    # An alert hit matches a ground truth event if the term matches event keywords AND alert day is within [start_day - lead_prior, start_day + lag_post]
    def get_matched_gt_event(row) -> Optional[str]:
        term = str(row["term"]).strip().lower()
        alert_day = parse_day_num(row["detected_window"])
        for _, er in gt_d21_30.iterrows():
            eid = er["event_id"]
            start_d = parse_day_num(er["start_day"])
            if (start_d - lead_prior) <= alert_day <= (start_d + lag_post):
                if term in gt_keywords.get(eid, []):
                    return eid
        return None

    base_d21_30["matched_event"] = base_d21_30.apply(get_matched_gt_event, axis=1)
    base_d21_30["is_tp_hit"] = base_d21_30["matched_event"].notna()
    base_tp = int(base_d21_30["is_tp_hit"].sum())
    base_fp = base_count - base_tp
    base_prec = base_tp / max(1, base_count)

    detected_by_baseline = set(base_d21_30["matched_event"].dropna().unique())
    base_rec = len(detected_by_baseline) / total_gt_events
    base_f1 = (2 * base_prec * base_rec / (base_prec + base_rec)) if (base_prec + base_rec) > 0 else 0.0

    # Baseline P@10 (sorted by surge_ratio descending)
    top10_base = base_d21_30.sort_values("surge_ratio", ascending=False).head(10)
    base_p10 = float(top10_base["is_tp_hit"].sum()) / max(1, len(top10_base))

    # 4. Pipeline Un-Gated (Phase 3 with 72h Rolling Buffer)
    if vetted_df is None:
        df_on, _ = run_pipeline_with_buffer(enabled=True)
        vetted_df = df_on[df_on["is_vetted_candidate"] == True].copy()
    else:
        vetted_df = vetted_df.copy()

    d21_30 = vetted_df[vetted_df["birth_window"].apply(lambda d: parse_day_num(d) >= 21)].copy()
    ungated_count = len(d21_30)

    d21_30["assigned_label"] = d21_30["trajectory_id"].astype(str).map(label_map).fillna("noise")
    ungated_tp = int((d21_30["assigned_label"] == "real").sum())
    ungated_fp = ungated_count - ungated_tp
    ungated_prec = ungated_tp / max(1, ungated_count)
    ungated_rec = ungated_tp / total_gt_events
    ungated_f1 = (2 * ungated_prec * ungated_rec / (ungated_prec + ungated_rec)) if (ungated_prec + ungated_rec) > 0 else 0.0

    top10_ungated = d21_30.sort_values("emergence_score", ascending=False).head(10)
    ungated_p10 = float((top10_ungated["assigned_label"] == "real").sum()) / max(1, len(top10_ungated))

    # 5. Pipeline Gated (Causal Novelty Gate @ sim_threshold from config)
    clusters_df = pd.read_csv(ROOT_DIR / "experiments" / "e6_combined" / "daily_clusters.csv")
    matches_df = pd.read_csv(ROOT_DIR / "experiments" / "e6_combined" / "daily_trajectory_matches.csv")
    embs_all = np.load(ROOT_DIR / "data" / "processed" / "daily" / "embeddings_daily.npy")

    cluster_centroids = {}
    for (win, cid), group in clusters_df.groupby(["window_id", "cluster_id"]):
        c_key = f"{win}_C{cid}"
        cent = np.mean(embs_all[group.index.values], axis=0)
        cent = cent / (np.linalg.norm(cent) + 1e-12)
        cluster_centroids[c_key] = (win, parse_day_num(win), cent)

    traj_centroids = {}
    for tid, group in matches_df.groupby("trajectory_id"):
        bm = group[group["is_birth"] == True]
        if len(bm) == 0:
            bm = group.head(1)
        b_key = bm.iloc[0]["cluster_key"]
        if b_key in cluster_centroids:
            traj_centroids[str(tid)] = cluster_centroids[b_key]

    gated_rows = []
    for _, r in d21_30.iterrows():
        tid = str(r["trajectory_id"])
        b_num = parse_day_num(r["birth_window"])
        cent = traj_centroids.get(tid, (None, None, None))[2]
        if cent is None:
            rep_p = str(r.get("representative_post", ""))
            pm = clusters_df[clusters_df["clean_text"] == rep_p]
            cent = embs_all[pm.index.values[0]] if len(pm) > 0 else np.zeros(384)
            if np.linalg.norm(cent) > 0:
                cent = cent / np.linalg.norm(cent)

        prior_cents = [c for (w, d, c) in cluster_centroids.values() if d < b_num]
        sims = np.dot(prior_cents, cent) if len(prior_cents) > 0 and np.linalg.norm(cent) > 0 else [0.0]
        max_sim = float(np.max(sims))
        if max_sim < sim_threshold:
            row_dict = r.to_dict()
            row_dict["max_prior_sim"] = max_sim
            row_dict["ranking_score"] = float(r.get("emergence_score", 0.0)) * (1.0 - 0.5 * max_sim)
            gated_rows.append(row_dict)

    gated_df = pd.DataFrame(gated_rows)
    gated_count = len(gated_df)
    gated_df["assigned_label"] = gated_df["trajectory_id"].astype(str).map(label_map).fillna("noise")
    gated_tp = int((gated_df["assigned_label"] == "real").sum())
    gated_fp = gated_count - gated_tp
    gated_prec = gated_tp / max(1, gated_count)
    gated_rec = gated_tp / total_gt_events
    gated_f1 = (2 * gated_prec * gated_rec / (gated_prec + gated_rec)) if (gated_prec + gated_rec) > 0 else 0.0

    top10_gated = gated_df.sort_values("ranking_score", ascending=False).head(10)
    gated_p10 = float((top10_gated["assigned_label"] == "real").sum()) / max(1, len(top10_gated))

    return {
        "baseline": {"count": base_count, "tp": base_tp, "fp": base_fp, "prec": base_prec, "rec": base_rec, "f1": base_f1, "p10": base_p10},
        "ungated": {"count": ungated_count, "tp": ungated_tp, "fp": ungated_fp, "prec": ungated_prec, "rec": ungated_rec, "f1": ungated_f1, "p10": ungated_p10},
        "gated": {"count": gated_count, "tp": gated_tp, "fp": gated_fp, "prec": gated_prec, "rec": gated_rec, "f1": gated_f1, "p10": gated_p10},
        "gt_count": total_gt_events,
        "quarantined": ungated_count - gated_count
    }


def step3_compute_empirical_benchmarks(vetted_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print(" STEP 3: EMPIRICAL BENCHMARK EVALUATION (DYNAMIC CALCULATION)")
    print("=" * 75)

    gt_path = ROOT_DIR / "data" / "ground_truth.csv"
    labels_path = ROOT_DIR / "review" / "alerts_to_label.csv"
    if not labels_path.exists():
        labels_path = ROOT_DIR / "review_public" / "alerts_to_label.csv"
    base_path = ROOT_DIR / "experiments" / "e6_combined" / "baseline_volume_alerts_daily.csv"

    if vetted_df is None:
        df_on, _ = run_pipeline_with_buffer(enabled=True)
        vetted_df = df_on[df_on["is_vetted_candidate"] == True].copy()

    res = compute_metrics_from_sources(gt_path, labels_path, base_path, vetted_df=vetted_df)
    b = res["baseline"]
    u = res["ungated"]
    g = res["gated"]

    print(f"\n{'System':<34} | {'Alerts':<6} | {'TP':<3} | {'FP':<4} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'P@10':<6}")
    print("-" * 90)
    print(f"{'Keyword Baseline':<34} | {b['count']:<6} | {b['tp']:<3} | {b['fp']:<4} | {b['prec']:<9.4f} | {b['rec']:<6.4f} | {b['f1']:<6.4f} | {b['p10']:<6.4f}")
    print(f"{'Phase 3 Pipeline (Un-Gated)':<34} | {u['count']:<6} | {u['tp']:<3} | {u['fp']:<4} | {u['prec']:<9.4f} | {u['rec']:<6.4f} | {u['f1']:<6.4f} | {u['p10']:<6.4f}")
    print(f"{'TrendRadar-AI (+ Novelty Gate)':<34} | {g['count']:<6} | {g['tp']:<3} | {g['fp']:<4} | {g['prec']:<9.4f} | {g['rec']:<6.4f} | {g['f1']:<6.4f} | {g['p10']:<6.4f}")
    print("-" * 90)
    print(f"[*] Ground Truth Recall Denominator: {res['gt_count']} real events on D21-D30")
    print(f"[*] Chatter Alerts Quarantined on D21-D30: {res['quarantined']} alerts ({(res['quarantined']/u['count'])*100:.1f}% reduction)")
    print("[+] Step 3 PASSED: Empirical benchmark dynamically computed with zero hard-coded values.")
    return vetted_df


def step3b_label_sensitivity_sanity_test(vetted_df: Optional[pd.DataFrame] = None):
    print("\n" + "=" * 75)
    print(" STEP 3B: LABEL SENSITIVITY SANITY TEST (TEMPORARY COPY MUTATION)")
    print("=" * 75)

    import tempfile
    import shutil

    gt_path = ROOT_DIR / "data" / "ground_truth.csv"
    real_labels_path = ROOT_DIR / "review" / "alerts_to_label.csv"
    if not real_labels_path.exists():
        real_labels_path = ROOT_DIR / "review_public" / "alerts_to_label.csv"
    base_path = ROOT_DIR / "experiments" / "e6_combined" / "baseline_volume_alerts_daily.csv"

    if vetted_df is None:
        df_on, _ = run_pipeline_with_buffer(enabled=True)
        vetted_df = df_on[df_on["is_vetted_candidate"] == True].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_labels_path = Path(tmpdir) / "temp_alerts_to_label.csv"
        shutil.copy(real_labels_path, temp_labels_path)

        # Baseline run on untouched labels
        orig_res = compute_metrics_from_sources(gt_path, temp_labels_path, base_path, vetted_df=vetted_df)
        orig_gated_tp = orig_res["gated"]["tp"]
        orig_gated_prec = orig_res["gated"]["prec"]
        orig_gated_rec = orig_res["gated"]["rec"]

        # Mutate one label in the temporary copy: change SEED_579 from 'real' to 'noise'
        temp_df = pd.read_csv(temp_labels_path)
        mask = temp_df["alert_id"] == "SEED_579"
        temp_df.loc[mask, "my_label"] = "noise"
        temp_df.to_csv(temp_labels_path, index=False)

        # Re-compute metrics on mutated copy
        mutated_res = compute_metrics_from_sources(gt_path, temp_labels_path, base_path, vetted_df=vetted_df)
        mut_gated_tp = mutated_res["gated"]["tp"]
        mut_gated_prec = mutated_res["gated"]["prec"]
        mut_gated_rec = mutated_res["gated"]["rec"]

        print(f"[*] Original Gated TP : {orig_gated_tp} (Precision: {orig_gated_prec:.4f}, Recall: {orig_gated_rec:.4f})")
        print(f"[*] Mutated Gated TP  : {mut_gated_tp} (Precision: {mut_gated_prec:.4f}, Recall: {mut_gated_rec:.4f})")

        # Verify dynamic change
        if mut_gated_tp == orig_gated_tp or mut_gated_prec == orig_gated_prec or mut_gated_rec == orig_gated_rec:
            raise RuntimeError("Sanity test FAILED: Metrics did not change when input label was mutated!")

        print("[+] Sensitivity Test PASSED: Metrics dynamically respond to input label changes.")

    # Confirm real file integrity
    real_check_df = pd.read_csv(real_labels_path)
    real_seed_label = real_check_df[real_check_df["alert_id"] == "SEED_579"].iloc[0]["my_label"]
    if real_seed_label != "real":
        raise RuntimeError("Integrity check FAILED: Real review file was modified!")
    print("[+] File Integrity Verified: Real review file remains untouched (SEED_579 label == 'real').")


def step4_run_pytest_suite():
    print("\n" + "=" * 75)
    print(" STEP 4: RUNNING AUTOMATED PYTEST SUITE (58 TESTS)")
    print("=" * 75)
    res = subprocess.run([sys.executable, "-m", "pytest", "-v"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("Pytest execution failed!")
    print("[+] Step 4 PASSED: All tests passed.")


def main():
    t0 = time.time()
    print("===========================================================================")
    print("   TRENDRADAR-AI: MASTER REPRODUCE-ALL AUTOMATED VERIFICATION")
    print("===========================================================================")
    step1_verify_pipeline_integrity()
    step2_run_shadow_replay()
    v_df = step3_compute_empirical_benchmarks()
    step3b_label_sensitivity_sanity_test(vetted_df=v_df)
    step4_run_pytest_suite()
    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f" MASTER REPRODUCTION COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("===========================================================================\n")


if __name__ == "__main__":
    main()
