"""
src/explainability.py
=====================
Deterministic Alert Evidence and Explanation Generator for TrendRadar-AI.
Constructs fully transparent mathematical justifications for alert firings without hallucination.
"""

from typing import Dict, Any, List, Optional
from src.config import get_config


def generate_alert_evidence(
    alert_data: Dict[str, Any],
    config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Constructs a deterministic, structured evidence dossier explaining why an alert fired.
    Compares observed metrics against configured emergence thresholds.
    """
    cfg = config or get_config()

    # Thresholds
    coherence_floor = float(getattr(cfg.emergence, "min_coherence_floor", 0.60) if hasattr(cfg, "emergence") else 0.60)
    velocity_thresh = float(getattr(cfg.emergence, "growth_ratio_threshold", 1.30) if hasattr(cfg, "emergence") else 1.30)
    novelty_thresh = float(getattr(cfg.novelty_gate, "similarity_threshold", 0.70) if hasattr(cfg, "novelty_gate") else 0.70)

    # Observed metrics
    tid = str(alert_data.get("trajectory_id", alert_data.get("id", "UNKNOWN")))
    day = str(alert_data.get("day", alert_data.get("window_id", alert_data.get("checkpoint_timestamp", "UNKNOWN"))))
    title = str(alert_data.get("title", f"Topic {tid}"))
    post_count = int(alert_data.get("post_count", alert_data.get("volume", 0)))
    coherence = float(alert_data.get("coherence", 0.0))
    velocity = float(alert_data.get("velocity", 0.0))
    acceleration = float(alert_data.get("acceleration", 0.0))
    emergence_score = float(alert_data.get("emergence_score", alert_data.get("score", 0.0)))
    novelty_score = float(alert_data.get("novelty_score", alert_data.get("max_similarity", 0.0)))
    novelty_status = str(alert_data.get("novelty_status", "passed"))

    kws = alert_data.get("keywords", alert_data.get("c_tf_idf_keywords", []))
    if isinstance(kws, str):
        keywords_list = [k.strip() for k in kws.replace(";", ",").split(",") if k.strip()]
    elif isinstance(kws, list):
        keywords_list = [str(k) for k in kws]
    else:
        keywords_list = []

    samples = alert_data.get("sample_posts", [alert_data.get("representative_post", "")])
    if isinstance(samples, str):
        samples_list = [samples]
    elif isinstance(samples, list):
        samples_list = [str(s) for s in samples if s]
    else:
        samples_list = []

    # Construct decision factors
    decision_factors: List[str] = []

    # 1. Volume Velocity Factor
    if velocity >= velocity_thresh:
        decision_factors.append(
            f"Volume velocity ({velocity:.2f}x) exceeded emergence threshold ({velocity_thresh:.2f}x), representing significant acceleration."
        )
    else:
        decision_factors.append(
            f"Volume velocity ({velocity:.2f}x) is below standard burst threshold ({velocity_thresh:.2f}x); candidate promoted via high coherence seed logic."
        )

    # 2. Semantic Coherence Factor
    if coherence >= coherence_floor:
        decision_factors.append(
            f"Dense semantic coherence ({coherence:.3f}) exceeded baseline floor ({coherence_floor:.3f}), confirming strong topical convergence."
        )
    else:
        decision_factors.append(
            f"Semantic coherence ({coherence:.3f}) is below standard floor ({coherence_floor:.3f})."
        )

    # 3. Novelty Gate Factor
    if novelty_score > 0.0:
        if novelty_score < novelty_thresh:
            decision_factors.append(
                f"Novelty gate verified novel discourse: maximum similarity to prior topic centroids ({novelty_score:.3f}) was below chatter quarantine threshold ({novelty_thresh:.3f})."
            )
        else:
            decision_factors.append(
                f"Novelty gate warning: maximum similarity to prior topic centroids ({novelty_score:.3f}) met or exceeded quarantine threshold ({novelty_thresh:.3f}) [{novelty_status}]."
            )
    else:
        decision_factors.append("Novelty gate: initial checkpoint emergence (no prior centroids exceeded similarity boundary).")

    # 4. Vocabulary Prominence Factor
    kw_str = ", ".join(keywords_list[:5]) if keywords_list else "None"
    decision_factors.append(f"Distinctive c-TF-IDF keyword anchor: [{kw_str}].")

    # Deterministic summary paragraph
    rep_quote = f'"{samples_list[0]}"' if samples_list else "N/A"
    summary_text = (
        f"Alert for #{tid} ('{title}') fired at checkpoint {day} with {post_count} posts. "
        f"The topic demonstrated velocity of {velocity:.2f}x and semantic coherence of {coherence:.3f} "
        f"(composite emergence score: {emergence_score:.2f}). "
        f"Dominant vocabulary terms include {kw_str}. "
        f"Representative community post: {rep_quote}."
    )

    return {
        "trajectory_id": tid,
        "day": day,
        "title": title,
        "metrics": {
            "post_count": post_count,
            "coherence": coherence,
            "velocity": velocity,
            "acceleration": acceleration,
            "emergence_score": emergence_score,
            "novelty_score": novelty_score,
            "novelty_status": novelty_status
        },
        "thresholds": {
            "coherence_floor": coherence_floor,
            "velocity_threshold": velocity_thresh,
            "novelty_threshold": novelty_thresh
        },
        "decision_factors": decision_factors,
        "keywords": keywords_list,
        "sample_posts": samples_list,
        "deterministic_explanation": summary_text
    }
