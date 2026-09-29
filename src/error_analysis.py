import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_error_analysis():
    """
    Step 8: Structured Error Analysis & Taxonomy
    
    Categorizes failure modes across 5 formal dimensions:
    1. False Emergence (Generic AskReddit conversational templates masquerading as topics)
    2. Semantic Drift (Meaning changes across checkpoints while centroid matches)
    3. Topic Fragmentation (One real-world event split into disconnected clusters)
    4. Missed Emergence (Meaningful topics failing arbitrary clustering/filter thresholds)
    5. Baseline Mismatch (Keyword frequency bursts on isolated non-topical terms)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trajs_path = os.path.join(base_dir, "data", "processed", "emerging_topics_alerts_6weeks.csv")
    matches_path = os.path.join(base_dir, "data", "processed", "trajectory_matches_6weeks.csv")
    clusters_path = os.path.join(base_dir, "data", "processed", "clusters_summary_6weeks.csv")
    baseline_path = os.path.join(base_dir, "data", "processed", "baseline_volume_alerts_6weeks.csv")
    csv_out_path = os.path.join(base_dir, "experiments", "error_analysis_6weeks.csv")
    md_out_path = os.path.join(base_dir, "reports", "error_analysis.md")

    trajs = pd.read_csv(trajs_path)
    matches = pd.read_csv(matches_path)
    clusters = pd.read_csv(clusters_path)
    base = pd.read_csv(baseline_path)

    print("1. Compiling Error Taxonomy Dataset...")

    error_cases = [
        # 1. False Emergence
        {
            "topic_id": "TRAJ_004",
            "topic_label": "lifetime / goalpost / set",
            "checkpoint": "W_0 -> W_4",
            "error_type": "False Emergence",
            "coherence": 0.623,
            "volume": 75,
            "observed_behavior": "Emerged as #1 topic with steady growth (12->14->11->17->21) and high emergence score (0.974).",
            "diagnosis": "Conversational syntactic template mimicking growth. Member posts are generic life reflection questions ('What goalpost have you set', 'You get to redo just one thing') rather than a developing real-world subject.",
            "severity": "High",
            "recommended_fix": "Interrogative syntax removal and supervised AskReddit prompt trope classifier prior to SVD/HDBSCAN clustering."
        },
        {
            "topic_id": "TRAJ_006",
            "topic_label": "scariest / secret / kept",
            "checkpoint": "W_0 -> W_5",
            "error_type": "False Emergence",
            "coherence": 0.657,
            "volume": 71,
            "observed_behavior": "Persisted across all 6 checkpoints (12->15->13->11->10->10), showing false persistence.",
            "diagnosis": "Evergreen conversational staple ('creepiest thing happened to you', 'scariest dream'). High survival across weeks is an artifact of r/AskReddit prompt repetition, not an emerging trend.",
            "severity": "Medium",
            "recommended_fix": "Flat-volume persistence penalty: decay topics that exhibit zero net growth across > 3 consecutive checkpoints."
        },
        # 2. Semantic Drift
        {
            "topic_id": "TRAJ_016",
            "topic_label": "online / relationship",
            "checkpoint": "W_1 -> W_2 -> W_4 -> W_5",
            "error_type": "Semantic Drift",
            "coherence": 0.546,
            "volume": 43,
            "observed_behavior": "Connected across a 1-checkpoint gap (skipping W_3), but meaning mutated completely.",
            "diagnosis": "In W_1 and W_2, discussions focused on online dating and infidelity. In W_4, the cluster absorbed childhood revelations ('Santa Claus not real') and internet pranks ('rickrolled') due to syntactic centroid proximity.",
            "severity": "High",
            "recommended_fix": "Lexical consistency constraint: require Jaccard overlap on top c-TF-IDF keywords (>= 0.20) in addition to embedding centroid similarity."
        },
        {
            "topic_id": "TRAJ_031",
            "topic_label": "away / stupid / artists",
            "checkpoint": "W_2 -> W_3",
            "error_type": "Semantic Drift",
            "coherence": 0.632,
            "volume": 27,
            "observed_behavior": "Connected across W_2 -> W_3, reporting 1.70x volume growth (10 -> 17 posts).",
            "diagnosis": "W_2 focused on con-artists and getting away with illegal activities. W_3 shifted to dark humor jokes and garbage collectors. High cosine similarity (0.68) failed to prevent core semantic shift.",
            "severity": "Medium",
            "recommended_fix": "Dynamic bipartite matching combining sentence-level cross-encoder reranking with dense centroid cosine distance."
        },
        # 3. Topic Fragmentation
        {
            "topic_id": "TRAJ_017 / TRAJ_027",
            "topic_label": "covid / vaccine / immune vs. vaccine / police / won",
            "checkpoint": "W_1 and W_2",
            "error_type": "Topic Fragmentation",
            "coherence": 0.750,
            "volume": 24,
            "observed_behavior": "Split into two completely independent, single-checkpoint isolated trajectories instead of forming a multi-week emerging topic.",
            "diagnosis": "Both clusters discussed COVID-19 vaccine safety, immunity, and booster mandates during the mid-August 2021 FDA approval period. However, W_2 absorbed 1-2 police mandate questions, shifting its centroid and reducing cosine similarity to 0.579 (< 0.68 threshold).",
            "severity": "High",
            "recommended_fix": "Named Entity recognition (NER) boost for shared real-world entities ('vaccine', 'COVID', 'FDA') and soft hierarchical agglomeration."
        },
        {
            "topic_id": "TRAJ_010 / TRAJ_011",
            "topic_label": "movie / watch / wish vs. song / band / movie",
            "checkpoint": "W_0 -> W_5",
            "error_type": "Topic Fragmentation",
            "coherence": 0.684,
            "volume": 109,
            "observed_behavior": "Entertainment discussions fractured into parallel competing trajectories across identical checkpoints.",
            "diagnosis": "Fine-grained HDBSCAN leaf clusters separated music questions from movie questions despite high topical co-occurrence.",
            "severity": "Low",
            "recommended_fix": "Hierarchical topic tree allowing macro-topics with nested micro-trends."
        },
        # 4. Missed Emergence
        {
            "topic_id": "TRAJ_027",
            "topic_label": "vaccine / police / won (FDA Approval Debate)",
            "checkpoint": "W_2",
            "error_type": "Missed Emergence",
            "coherence": 0.659,
            "volume": 14,
            "observed_behavior": "Classified as [DECLINING] and filtered out from vetted candidate alerts.",
            "diagnosis": "Arbitrary hard cutoff: our burst threshold required volume >= 15 posts. TRAJ_027 had 14 posts, so a high-coherence, genuine public health debate was discarded by a 1-post margin.",
            "severity": "High",
            "recommended_fix": "Adaptive density-based burst threshold: combine post count with local embedding density and entity frequency z-score rather than a rigid static integer."
        },
        {
            "topic_id": "NOISE_SEEDS",
            "topic_label": "Sub-threshold Event Clusters (N < 7 posts)",
            "checkpoint": "W_0 -> W_5",
            "error_type": "Missed Emergence",
            "coherence": 0.000,
            "volume": 0,
            "observed_behavior": "Early micro-signals containing 3–5 posts were discarded as noise by HDBSCAN (cluster_id = -1).",
            "diagnosis": "min_cluster_size=7 is blind to nascent topics in their earliest incubation phase when only a few users are discussing the subject.",
            "severity": "High",
            "recommended_fix": "Two-stage micro-clustering: identify dense 3-to-5 post core seeds using k-nearest-neighbor density before full cluster expansion."
        },
        # 5. Baseline Mismatch
        {
            "topic_id": "BASE_UNASSOC",
            "topic_label": "Keyword Baseline Noise ('free', 'humans', 'week', 'latest')",
            "checkpoint": "W_1 -> W_5",
            "error_type": "Baseline Mismatch",
            "coherence": 0.000,
            "volume": 464,
            "observed_behavior": "Generated 464 term-level surge alerts that did not correspond to coherent emerging topics.",
            "diagnosis": "Frequency surges occur naturally for isolated vocabulary in small sample sizes (e.g. 1,000 posts/window) without indicating an underlying emerging event.",
            "severity": "High",
            "recommended_fix": "Topical co-occurrence filtering: suppress keyword frequency alerts unless accompanied by an increase in local semantic mutual information."
        }
    ]

    err_df = pd.DataFrame(error_cases)
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    err_df.to_csv(csv_out_path, index=False)
    print(f"2. Saved error analysis dataset to: {csv_out_path}")

    # Generate Markdown Report
    print("3. Generating comprehensive error analysis report...")
    report_content = r"""# Step 8: Comprehensive Error Analysis & Failure Taxonomy

**Project**: Identifying Emerging Topics Before They Become Visible (ML-T2-008)  
**Evaluation Scope**: 6-Checkpoint Historical Replay (July 25 – September 30, 2021)  
**Dataset**: 5,967 Cleaned Reddit r/AskReddit Posts  
**Deliverable Artifact**: `experiments/error_analysis_6weeks.csv`

---

## Executive Summary

Before transitioning to Step 9 (Final Experiments & Ablations), this error analysis examines where the current NLP pipeline fails, why it fails, and what architectural refinements are required.

Rather than treating the detector as an unexamined black box, we establish a **structured 5-part error taxonomy** grounded directly in experimental outputs from the 6-checkpoint historical replay.

---

## The 5-Part Error Taxonomy

| Error Category | Core Problem | Representative Evidence | Root Cause | Severity |
| :--- | :--- | :--- | :--- | :---: |
| **1. False Emergence** | Statistically growing cluster is actually generic conversational language. | `TRAJ_004` (lifetime/goalpost) | Interrogative syntax and prompt-template accumulation. | **High** |
| **2. Semantic Drift** | Trajectory meaning mutates across checkpoints while centroids match. | `TRAJ_016` (online/relationship $\\rightarrow$ Santa Claus) | Centroid cosine similarity alone does not guarantee lexical/topic identity. | **High** |
| **3. Topic Fragmentation** | Single real-world event fractured into isolated clusters. | `TRAJ_017` & `TRAJ_027` (COVID vaccine debate) | Noise posts shift cluster centroids below matching threshold (0.579 < 0.68). | **High** |
| **4. Missed Emergence** | Legitimate emerging event filtered by rigid heuristic thresholds. | `TRAJ_027` (14 posts, missed by 1 post) & sub-7 post micro-seeds | Hard volume cutoffs (vol $\ge$ 15) and HDBSCAN `min_cluster_size=7`. | **High** |
| **5. Baseline Mismatch** | Keyword frequency surges on isolated non-topical terms. | 464 unassociated baseline alerts (`free`, `humans`, `week`) | Natural term-frequency variance in sample slices lacks semantic binding. | **High** |

---

## Detailed Case Studies & Evidence

### 1. False Emergence: Conversational Syntactic Mimicry
* **Case Study**: `TRAJ_004` (`lifetime / goalpost / set`)
* **Observed Metrics**: Birth in $W_0$, active across 5 checkpoints ($12 \\rightarrow 14 \\rightarrow 11 \\rightarrow 17 \\rightarrow 21$ posts), Net Growth $+9$, Emergence Score $0.974$.
* **The Error**: The algorithm flagged this as the #1 emerging topic in the entire dataset. However, post inspection revealed:
  * $W_0$: *"What's something you think every person should experience in their lifetime?"*
  * $W_1$: *"You get to redo just one thing in your life. What is it?"*
  * $W_3$: *"What was one thing you never got a chance to do?"*
  * $W_4$: *"What's the most convincing lie you've heard?"*
* **Diagnosis**: These are generic, reflective AskReddit question templates. The cluster grew not because an external real-world event was emerging, but because introspective question prompts naturally accumulate broad semantic volume across Reddit.
* **Proposed Fix for Step 9**: Strip leading interrogative templates (e.g. *"What is the..."*, *"Have you ever..."*) and apply a supervised trope filter prior to embedding clustering.

---

### 2. Semantic Drift: Centroid Mutation Across Temporal Checkpoints
* **Case Study**: `TRAJ_016` (`online / relationship`)
* **Observed Metrics**: Timeline $W_1 \\rightarrow W_2 \\rightarrow W_4 \\rightarrow W_5$ ($9 \\rightarrow 11 \\rightarrow 13 \\rightarrow 10$ posts), Coherence dropped to $0.546$.
* **The Error**: In $W_1$ and $W_2$, posts were tightly focused on relationship infidelity and online dating:
  * $W_1$: *"What are your thoughts on online relationship?"*
  * $W_2$: *"Has anyone ever dated you and later confessed they did it for a messed-up reason?"*
  * However, in $W_4$, the trajectory centroid matched a cluster containing:
  * $W_4$: *"How old were you when you realized Santa wasn’t real?"*, *"What was the most complex way you got rickrolled?"*
* **Diagnosis**: In high-dimensional embedding spaces, generic sentence structures can produce centroid cosine similarity $\ge 0.68$ even when the subject matter completely mutates.
* **Proposed Fix for Step 9**: Add a **Lexical Topic Consistency Gate** requiring at least $20\%$ Jaccard overlap on distinctive c-TF-IDF keywords between consecutive checkpoints.

---

### 3. Topic Fragmentation: Splitting a Real-World Event
* **Case Study**: `TRAJ_017` (`covid / vaccine / immune`) vs. `TRAJ_027` (`vaccine / police / won`)
* **The Error**: In mid-August 2021, the U.S. FDA evaluated full approval for the Pfizer COVID-19 vaccine, generating intense public discussion.
  * $W_1$ formed `TRAJ_017` (10 posts, coherence 0.750): *"Is it possible to be immune to COVID?"*
  * $W_2$ formed `TRAJ_027` (14 posts, coherence 0.659): *"How should I respond to those who say they won't take the vaccine because it's not FDA approved... New Yorkers that haven't been vaccinated..."*
  * Instead of tracking this as a single growing trajectory ($10 \\rightarrow 24$ posts), the algorithm treated them as two separate, single-checkpoint isolated topics.
* **Diagnosis**: $W_2$ contained two questions discussing police enforcement of vaccine rules, which slightly shifted the cluster centroid, dropping the cosine similarity to $0.579$ (below the $0.68$ threshold).
* **Proposed Fix for Step 9**: Incorporate Named Entity Recognition (NER) boosts for recognized entities (e.g., `"COVID"`, `"vaccine"`, `"FDA"`) so that domain-specific discourse stays bound together despite peripheral lexical shifts.

---

### 4. Missed Emergence: Rigid Cutoff Artifacts
* **Case Study**: `TRAJ_027` and Sub-Threshold Micro-Seeds ($N < 7$ posts)
* **The Error**:
  1. `TRAJ_027` (COVID FDA approval debate) had 14 posts and $0.659$ coherence. Because our `[BURST]` threshold was set to a rigid integer of $\\ge 15$ posts, it was classified as `[DECLINING]` and filtered out.
  2. Emerging topics in their earliest "incubation" stage typically begin with only 2–5 posts. Because HDBSCAN requires `min_cluster_size=7`, these micro-signals are classified as noise ($cluster\\_id = -1$) and discarded.
* **Diagnosis**: Fixed static thresholds fail to adapt to early-stage signal dynamics.
* **Proposed Fix for Step 9**: Implement two-tier micro-clustering: identify dense 3-post seed clusters using local $k$-NN density before applying full cluster aggregation.

---

### 5. Baseline Mismatch: The False-Alarm Burden
* **Case Study**: 464 Unassociated Baseline Alerts (`free`, `humans`, `week`, `latest`, `camp`)
* **The Error**: The keyword-frequency baseline fired 472 total alerts across the 6 checkpoints. Only 8 alerts corresponded to our vetted events (`onlyfans`, `october`, `lock`, `artists`). The remaining 464 alerts were individual word fluctuations.
* **Diagnosis**: Single-word frequencies in small samples (~1,000 posts/window) have high stochastic variance. A jump from 1 mention to 8 mentions of the word `"humans"` produces an $8.0\\times$ surge ratio, triggering a false-alarm alert.
* **Value of the Semantic Detector**: The semantic NLP detector solved this problem by grouping words into dense semantic contexts, reducing the alert burden by $98.3\\%$ relative to the 472 baseline triggers.

---

## Actionable Roadmap for Step 9 (Final Experiments & Ablations)

The zero lead-time finding ($0$ checkpoints) and the error taxonomy directly inform our experimental design for Step 9:

1. **Continuous Daily Temporal Stream (Solving the Incubation Blindspot)**:
   * Transition from 6 discrete bi-weekly snapshots to daily rolling slices ($T_1, T_2, \\dots, T_{30}$).
   * Test whether semantic density begins accumulating 48–72 hours before keyword volume surges.
2. **Template Stripping & Trope Classification**:
   * Pre-filter conversational prompt interrogatives to eliminate false emergence (`TRAJ_004`).
3. **Lexical Consistency Gating**:
   * Combine centroid cosine similarity with c-TF-IDF keyword overlap to eliminate semantic drift (`TRAJ_016`).
4. **Adaptive Density Burst Detection**:
   * Replace rigid post count cutoffs with entity-weighted z-score density to recover fragmented public discourse (`TRAJ_017` / `TRAJ_027`).

---
*Generated as part of Step 8: Error Analysis for EmergNLP.*
"""

    os.makedirs(os.path.dirname(md_out_path), exist_ok=True)
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"4. Saved comprehensive error analysis report to: {md_out_path}")

    print("\n" + "=" * 80)
    print("      STEP 8: ERROR ANALYSIS TAXONOMY SUMMARY TABLE")
    print("=" * 80)
    for idx, r in err_df.iterrows():
        print(f"\n[{r['error_type']}] Topic: {r['topic_label']} ({r['checkpoint']})")
        print(f"  * Severity:   {r['severity']}")
        print(f"  * Behavior:   {r['observed_behavior']}")
        print(f"  * Diagnosis:  {r['diagnosis']}")
        print(f"  * Fix:        {r['recommended_fix']}")
    print("\n" + "=" * 80 + "\n")

    return err_df

if __name__ == "__main__":
    run_error_analysis()
