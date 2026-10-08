from pathlib import Path
import sys
import re
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.config import ROOT_DIR, get_config

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = ROOT_DIR


def load_config(config_path: Optional[Any] = None) -> Dict[str, Any]:
    """Loads settings from config.yaml or provides robust defaults."""
    cfg = get_config(config_path=Path(config_path) if config_path else None)
    return cfg.to_dict()



def parse_day_num(day_val: Any) -> int:
    """
    Extracts chronological integer index from a day identifier.
    Supports formats like 'D_01' (1), 'D11' (11), 'W_2' (2), or integer 11.
    """
    if pd.isna(day_val):
        return -1
    match = re.search(r'\d+', str(day_val))
    return int(match.group()) if match else -1


def get_split_name(day_val: Any, tuning_days: List[str], held_out_days: List[str]) -> str:
    """Categorizes a day into 'tuning', 'held_out', or 'other' based on config."""
    d_str = str(day_val).strip()
    if d_str in tuning_days:
        return "tuning"
    if d_str in held_out_days:
        return "held_out"

    d_num = parse_day_num(d_str)
    tuning_nums = [parse_day_num(x) for x in tuning_days]
    held_out_nums = [parse_day_num(x) for x in held_out_days]

    if d_num in tuning_nums or (1 <= d_num <= 20):
        return "tuning"
    if d_num in held_out_nums or (21 <= d_num <= 30):
        return "held_out"
    return "other"


def match_keywords(text: str, keywords: List[str]) -> List[str]:
    """Finds which ground-truth keywords are present in the text."""
    text_clean = str(text).lower()
    matched = []
    for kw in keywords:
        kw_clean = kw.strip().lower()
        if not kw_clean:
            continue
        pattern = r'\b' + re.escape(kw_clean) + r'\b'
        if re.search(pattern, text_clean):
            matched.append(kw_clean)
    return matched


def match_alert_to_events(
    alert_day_num: int,
    alert_text: str,
    gt_df: pd.DataFrame,
    lead_prior: int,
    lag_post: int,
    min_keyword_matches: int
) -> List[Dict[str, Any]]:
    """
    Checks if an alert matches any ground-truth event based on keywords and day window.
    An alert at day t_alert matches event at start_day if:
    (start_day - lead_prior) <= t_alert <= (start_day + lag_post)
    and keyword overlap >= min_keyword_matches.
    """
    matches = []
    if alert_day_num < 0:
        return matches

    for _, event_row in gt_df.iterrows():
        event_day_num = parse_day_num(event_row["start_day"])
        if event_day_num < 0:
            continue

        # Check temporal window
        if not ((event_day_num - lead_prior) <= alert_day_num <= (event_day_num + lag_post)):
            continue

        # Check keyword matches
        event_kws = [k.strip().lower() for k in str(event_row["keywords"]).split(",") if k.strip()]
        matched_kws = match_keywords(alert_text, event_kws)

        if len(matched_kws) >= min_keyword_matches:
            lead_time_days = event_day_num - alert_day_num
            matches.append({
                "event_id": event_row["event_id"],
                "label": event_row["label"],
                "event_start_day": event_row["start_day"],
                "lead_time_days": lead_time_days,
                "matched_keywords": matched_kws
            })

    return matches


def compute_metrics(
    alerts_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    alert_type: str = "pipeline",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1, Precision@10, and Mean Lead Time
    against annotated ground-truth events.
    """
    eval_cfg = config.get("evaluation", {}) if config else {}
    lead_prior = int(eval_cfg.get("lead_window_days_prior", 2))
    lag_post = int(eval_cfg.get("lag_window_days_post", 3))
    min_kws = int(eval_cfg.get("min_keyword_matches", 1))

    real_events = gt_df[gt_df["label"] == "real_event"]
    total_real_events = len(real_events)

    if alerts_df.empty:
        return {
            "total_alerts": 0,
            "true_positives": 0,
            "false_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "precision_at_10": 0.0,
            "detected_real_events": 0,
            "total_real_events": total_real_events,
            "mean_lead_time_days": 0.0,
            "detected_event_ids": []
        }

    # Sort alerts for Precision@10 ranking
    if alert_type == "pipeline":
        sort_col = "emergence_score" if "emergence_score" in alerts_df.columns else "total_volume"
        sorted_alerts = alerts_df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    else:
        sort_col = "surge_ratio" if "surge_ratio" in alerts_df.columns else "current_volume"
        sorted_alerts = alerts_df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    true_positives = 0
    false_positives = 0
    event_lead_times: Dict[str, int] = {}
    detected_event_ids = set()

    for _, a_row in sorted_alerts.iterrows():
        if alert_type == "pipeline":
            alert_day = a_row.get("birth_window", a_row.get("detected_window", ""))
            alert_text = f"{a_row.get('topic_name', '')} {a_row.get('representative_post', '')}"
        else:
            alert_day = a_row.get("detected_window", "")
            alert_text = str(a_row.get("term", ""))

        day_num = parse_day_num(alert_day)
        matches = match_alert_to_events(day_num, alert_text, gt_df, lead_prior, lag_post, min_kws)

        # Check if matched to any real event
        real_matches = [m for m in matches if m["label"] == "real_event"]
        if real_matches:
            true_positives += 1
            for m in real_matches:
                eid = m["event_id"]
                detected_event_ids.add(eid)
                lead_t = m["lead_time_days"]
                # Keep earliest detection (maximum lead time)
                if eid not in event_lead_times or lead_t > event_lead_times[eid]:
                    event_lead_times[eid] = lead_t
        else:
            false_positives += 1

    total_alerts = len(sorted_alerts)
    precision = (true_positives / total_alerts) if total_alerts > 0 else 0.0
    recall = (len(detected_event_ids) / total_real_events) if total_real_events > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall + 1e-9)) if (precision + recall) > 0 else 0.0

    # Precision @ 10
    top_10 = sorted_alerts.head(10)
    top_10_tp = 0
    for _, a_row in top_10.iterrows():
        if alert_type == "pipeline":
            alert_day = a_row.get("birth_window", a_row.get("detected_window", ""))
            alert_text = f"{a_row.get('topic_name', '')} {a_row.get('representative_post', '')}"
        else:
            alert_day = a_row.get("detected_window", "")
            alert_text = str(a_row.get("term", ""))

        day_num = parse_day_num(alert_day)
        matches = match_alert_to_events(day_num, alert_text, gt_df, lead_prior, lag_post, min_kws)
        if any(m["label"] == "real_event" for m in matches):
            top_10_tp += 1

    precision_at_10 = top_10_tp / min(10, len(top_10)) if len(top_10) > 0 else 0.0

    # Mean lead time across detected real events
    mean_lead_time = float(np.mean(list(event_lead_times.values()))) if event_lead_times else 0.0

    return {
        "total_alerts": total_alerts,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "precision_at_10": round(float(precision_at_10), 4),
        "detected_real_events": len(detected_event_ids),
        "total_real_events": total_real_events,
        "mean_lead_time_days": round(float(mean_lead_time), 2),
        "detected_event_ids": sorted(list(detected_event_ids))
    }


def evaluate_pipeline(
    ground_truth_path: Optional[str] = None,
    candidates_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
    config_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
    output_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates both TrendRadar-AI and Keyword Volume Baseline against ground truth.
    Splits evaluation into:
      - Tuning Set: D00 to D19 (Days 1 to 20)
      - Held-out Test Set: D20 to D29 (Days 21 to 30)
      - Full Stream (Overall)
    """
    cfg = load_config(config_path)
    eval_cfg = cfg.get("evaluation", {})
    tuning_days = eval_cfg.get("tuning_days", [f"D_{i:02d}" for i in range(1, 21)])
    held_out_days = eval_cfg.get("held_out_days", [f"D_{i:02d}" for i in range(21, 31)])

    if ground_truth_path is None:
        ground_truth_path = BASE_DIR / "data" / "ground_truth.csv"
    else:
        ground_truth_path = Path(ground_truth_path)

    if candidates_path is None:
        p1 = BASE_DIR / "experiments" / "e6_combined" / "emergence_alerts.csv"
        p2 = BASE_DIR / "data" / "processed" / "emerging_topics_alerts_6weeks.csv"
        candidates_path = p1 if p1.exists() else p2
    else:
        candidates_path = Path(candidates_path)

    if baseline_path is None:
        b1 = BASE_DIR / "experiments" / "e6_combined" / "baseline_volume_alerts_daily.csv"
        b2 = BASE_DIR / "data" / "processed" / "baseline_volume_alerts_6weeks.csv"
        baseline_path = b1 if b1.exists() else b2
    else:
        baseline_path = Path(baseline_path)

    if output_json_path is None:
        output_json_path = BASE_DIR / "experiments" / "eval_results.json"
    else:
        output_json_path = Path(output_json_path)

    if output_csv_path is None:
        output_csv_path = BASE_DIR / "experiments" / "evaluation_results_6weeks.csv"
    else:
        output_csv_path = Path(output_csv_path)

    print(f"Loading Ground Truth Annotations from: {ground_truth_path}")
    gt_df = pd.read_csv(ground_truth_path)

    print(f"Loading Pipeline Candidates from: {candidates_path}")
    cand_df = pd.read_csv(candidates_path)
    # Consider candidate alerts: either vetted candidates or positive emergence score
    if "is_vetted_candidate" in cand_df.columns:
        cand_alerts = cand_df[cand_df["is_vetted_candidate"] == True].copy()
    else:
        cand_alerts = cand_df.copy()

    print(f"Loading Keyword Baseline Alerts from: {baseline_path}")
    base_df = pd.read_csv(baseline_path)

    # Annotate split on each table
    gt_df["split"] = gt_df["start_day"].apply(lambda d: get_split_name(d, tuning_days, held_out_days))

    cand_day_col = "birth_window" if "birth_window" in cand_alerts.columns else "detected_window"
    cand_alerts["split"] = cand_alerts[cand_day_col].apply(lambda d: get_split_name(d, tuning_days, held_out_days))

    base_day_col = "detected_window" if "detected_window" in base_df.columns else "window_id"
    base_df["split"] = base_df[base_day_col].apply(lambda d: get_split_name(d, tuning_days, held_out_days))

    # Split GT and alerts
    gt_tuning = gt_df[gt_df["split"] == "tuning"]
    gt_held_out = gt_df[gt_df["split"] == "held_out"]

    cand_tuning = cand_alerts[cand_alerts["split"] == "tuning"]
    cand_held_out = cand_alerts[cand_alerts["split"] == "held_out"]

    base_tuning = base_df[base_df["split"] == "tuning"]
    base_held_out = base_df[base_df["split"] == "held_out"]

    # Compute Metrics across Splits
    print("Computing metrics for Pipeline & Baseline across Tuning and Held-out Test...")
    pipeline_full = compute_metrics(cand_alerts, gt_df, alert_type="pipeline", config=cfg)
    pipeline_tuning = compute_metrics(cand_tuning, gt_tuning, alert_type="pipeline", config=cfg)
    pipeline_held_out = compute_metrics(cand_held_out, gt_held_out, alert_type="pipeline", config=cfg)

    baseline_full = compute_metrics(base_df, gt_df, alert_type="baseline", config=cfg)
    baseline_tuning = compute_metrics(base_tuning, gt_tuning, alert_type="baseline", config=cfg)
    baseline_held_out = compute_metrics(base_held_out, gt_held_out, alert_type="baseline", config=cfg)

    # Detailed Event-by-Event Audit
    lead_prior = int(eval_cfg.get("lead_window_days_prior", 2))
    lag_post = int(eval_cfg.get("lag_window_days_post", 3))
    min_kws = int(eval_cfg.get("min_keyword_matches", 1))

    event_audit = []
    comparison_rows = []

    for _, e_row in gt_df.iterrows():
        eid = e_row["event_id"]
        label = e_row["label"]
        start_day = e_row["start_day"]
        day_num = parse_day_num(start_day)
        split = e_row["split"]
        notes = e_row.get("notes", "")
        e_kws = [k.strip().lower() for k in str(e_row["keywords"]).split(",") if k.strip()]

        # Check Pipeline Matches
        p_matches = []
        for _, c_row in cand_alerts.iterrows():
            c_day = c_row.get(cand_day_col, "")
            c_day_num = parse_day_num(c_day)
            if (day_num - lead_prior) <= c_day_num <= (day_num + lag_post):
                c_text = f"{c_row.get('topic_name', '')} {c_row.get('representative_post', '')}"
                matched_kws = match_keywords(c_text, e_kws)
                if len(matched_kws) >= min_kws:
                    p_matches.append({
                        "trajectory_id": c_row.get("trajectory_id", ""),
                        "detection_day": c_day,
                        "day_num": c_day_num,
                        "lead_time": day_num - c_day_num,
                        "matched_kws": matched_kws,
                        "topic_name": c_row.get("topic_name", ""),
                        "representative_post": c_row.get("representative_post", "")
                    })

        # Check Baseline Matches
        b_matches = []
        for _, b_row in base_df.iterrows():
            b_day = b_row.get(base_day_col, "")
            b_day_num = parse_day_num(b_day)
            if (day_num - lead_prior) <= b_day_num <= (day_num + lag_post):
                b_term = str(b_row.get("term", "")).lower()
                matched_kws = [kw for kw in e_kws if (kw == b_term or kw in b_term or b_term in kw)]
                if len(matched_kws) >= min_kws:
                    b_matches.append({
                        "term": b_term,
                        "detection_day": b_day,
                        "day_num": b_day_num,
                        "lead_time": day_num - b_day_num,
                        "surge_ratio": b_row.get("surge_ratio", 0)
                    })

        p_detected = len(p_matches) > 0
        b_detected = len(b_matches) > 0

        p_lead_time = max([m["lead_time"] for m in p_matches]) if p_matches else None
        b_lead_time = max([m["lead_time"] for m in b_matches]) if b_matches else None

        earliest_p_day = p_matches[0]["detection_day"] if p_matches else "NEVER"
        earliest_b_day = b_matches[0]["detection_day"] if b_matches else "NEVER"

        event_audit.append({
            "event_id": eid,
            "label": label,
            "start_day": start_day,
            "split": split,
            "notes": notes,
            "pipeline_detected": p_detected,
            "pipeline_earliest_day": earliest_p_day,
            "pipeline_lead_time_days": p_lead_time,
            "pipeline_matched_count": len(p_matches),
            "baseline_detected": b_detected,
            "baseline_earliest_day": earliest_b_day,
            "baseline_lead_time_days": b_lead_time,
            "baseline_matched_count": len(b_matches)
        })

        if p_matches:
            best_p = p_matches[0]
            comparison_rows.append({
                "candidate_id": best_p["trajectory_id"],
                "topic_label": str(best_p["topic_name"]),
                "candidate_type": label,
                "nlp_detection_window": best_p["detection_day"],
                "nlp_window_index": best_p["day_num"],
                "baseline_detection_window": earliest_b_day,
                "baseline_window_index": b_matches[0]["day_num"] if b_matches else None,
                "lead_time_checkpoints": p_lead_time if p_lead_time is not None else "N/A",
                "baseline_matched_terms": ", ".join([m["term"] for m in b_matches[:4]]) if b_matches else "None",
                "baseline_alert_count": len(b_matches),
                "representative_sample": best_p.get("representative_post", "")[:120]
            })

    results = {
        "metadata": {
            "tuning_days": tuning_days,
            "held_out_days": held_out_days,
            "total_ground_truth_events": len(gt_df),
            "real_events_count": int((gt_df["label"] == "real_event").sum()),
            "noise_events_count": int((gt_df["label"] == "noise").sum())
        },
        "pipeline": {
            "full_stream": pipeline_full,
            "tuning_split": pipeline_tuning,
            "held_out_test_split": pipeline_held_out
        },
        "baseline": {
            "full_stream": baseline_full,
            "tuning_split": baseline_tuning,
            "held_out_test_split": baseline_held_out
        },
        "event_detection_audit": event_audit
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation JSON results to: {output_json_path}")

    # Save CSV for Streamlit app backwards compatibility
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(output_csv_path, index=False)
        print(f"Saved dashboard compatibility CSV to: {output_csv_path}")

    # Print Senior-Engineer Summary
    print("\n" + "=" * 80)
    print("          RIGOROUS OBJECTIVE EVALUATION: PIPELINE vs BASELINE")
    print("=" * 80)
    print(f"Splits: Tuning (D00-D19, {len(tuning_days)} days) | Held-Out Test (D20-D29, {len(held_out_days)} days)")
    print(f"Ground Truth: {len(gt_df)} annotated events ({results['metadata']['real_events_count']} real, {results['metadata']['noise_events_count']} noise)\n")

    summary_df = pd.DataFrame([
        {
            "System": "TrendRadar Pipeline",
            "Split": "Tuning (D00-D19)",
            "Alerts": pipeline_tuning["total_alerts"],
            "Prec": f"{pipeline_tuning['precision']:.3f}",
            "Recall": f"{pipeline_tuning['recall']:.3f}",
            "F1": f"{pipeline_tuning['f1']:.3f}",
            "P@10": f"{pipeline_tuning['precision_at_10']:.3f}",
            "Lead Time (d)": f"{pipeline_tuning['mean_lead_time_days']:.1f}"
        },
        {
            "System": "TrendRadar Pipeline",
            "Split": "Held-Out Test (D20-D29)",
            "Alerts": pipeline_held_out["total_alerts"],
            "Prec": f"{pipeline_held_out['precision']:.3f}",
            "Recall": f"{pipeline_held_out['recall']:.3f}",
            "F1": f"{pipeline_held_out['f1']:.3f}",
            "P@10": f"{pipeline_held_out['precision_at_10']:.3f}",
            "Lead Time (d)": f"{pipeline_held_out['mean_lead_time_days']:.1f}"
        },
        {
            "System": "Keyword Surge Baseline",
            "Split": "Tuning (D00-D19)",
            "Alerts": baseline_tuning["total_alerts"],
            "Prec": f"{baseline_tuning['precision']:.3f}",
            "Recall": f"{baseline_tuning['recall']:.3f}",
            "F1": f"{baseline_tuning['f1']:.3f}",
            "P@10": f"{baseline_tuning['precision_at_10']:.3f}",
            "Lead Time (d)": f"{baseline_tuning['mean_lead_time_days']:.1f}"
        },
        {
            "System": "Keyword Surge Baseline",
            "Split": "Held-Out Test (D20-D29)",
            "Alerts": baseline_held_out["total_alerts"],
            "Prec": f"{baseline_held_out['precision']:.3f}",
            "Recall": f"{baseline_held_out['recall']:.3f}",
            "F1": f"{baseline_held_out['f1']:.3f}",
            "P@10": f"{baseline_held_out['precision_at_10']:.3f}",
            "Lead Time (d)": f"{baseline_held_out['mean_lead_time_days']:.1f}"
        }
    ])
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    return results


# Backwards compatibility function
def evaluate_early_detection(
    candidates_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Wrapper ensuring backward compatibility with existing scripts/dashboard."""
    res = evaluate_pipeline(
        candidates_path=candidates_path,
        baseline_path=baseline_path,
        output_csv_path=output_path
    )
    if output_path and os.path.exists(output_path):
        return pd.read_csv(output_path)
    return pd.DataFrame(res.get("event_detection_audit", []))


if __name__ == "__main__":
    evaluate_pipeline()
