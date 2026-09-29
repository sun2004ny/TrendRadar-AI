# Step 8: Comprehensive Error Analysis & Failure Taxonomy

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
| **2. Semantic Drift** | Trajectory meaning mutates across checkpoints while centroids match. | `TRAJ_016` (online/relationship $\rightarrow$ Santa Claus) | Centroid cosine similarity alone does not guarantee lexical/topic identity. | **High** |
| **3. Topic Fragmentation** | Single real-world event fractured into isolated clusters. | `TRAJ_017` & `TRAJ_027` (COVID vaccine debate) | Noise posts shift cluster centroids below matching threshold (0.579 < 0.68). | **High** |
| **4. Missed Emergence** | Legitimate emerging event filtered by rigid heuristic thresholds. | `TRAJ_027` (14 posts, missed by 1 post) & sub-7 post micro-seeds | Hard volume cutoffs (vol $\ge$ 15) and HDBSCAN `min_cluster_size=7`. | **High** |
| **5. Baseline Mismatch** | Keyword frequency surges on isolated non-topical terms. | 464 unassociated baseline alerts (`free`, `humans`, `week`) | Natural term-frequency variance in sample slices lacks semantic binding. | **High** |

---

## Detailed Case Studies & Evidence

### 1. False Emergence: Conversational Syntactic Mimicry
* **Case Study**: `TRAJ_004` (`lifetime / goalpost / set`)
* **Observed Metrics**: Birth in $W_0$, active across 5 checkpoints ($12 \rightarrow 14 \rightarrow 11 \rightarrow 17 \rightarrow 21$ posts), Net Growth $+9$, Emergence Score $0.974$.
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
* **Observed Metrics**: Timeline $W_1 \rightarrow W_2 \rightarrow W_4 \rightarrow W_5$ ($9 \rightarrow 11 \rightarrow 13 \rightarrow 10$ posts), Coherence dropped to $0.546$.
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
  * Instead of tracking this as a single growing trajectory ($10 \rightarrow 24$ posts), the algorithm treated them as two separate, single-checkpoint isolated topics.
* **Diagnosis**: $W_2$ contained two questions discussing police enforcement of vaccine rules, which slightly shifted the cluster centroid, dropping the cosine similarity to $0.579$ (below the $0.68$ threshold).
* **Proposed Fix for Step 9**: Incorporate Named Entity Recognition (NER) boosts for recognized entities (e.g., `"COVID"`, `"vaccine"`, `"FDA"`) so that domain-specific discourse stays bound together despite peripheral lexical shifts.

---

### 4. Missed Emergence: Rigid Cutoff Artifacts
* **Case Study**: `TRAJ_027` and Sub-Threshold Micro-Seeds ($N < 7$ posts)
* **The Error**:
  1. `TRAJ_027` (COVID FDA approval debate) had 14 posts and $0.659$ coherence. Because our `[BURST]` threshold was set to a rigid integer of $\ge 15$ posts, it was classified as `[DECLINING]` and filtered out.
  2. Emerging topics in their earliest "incubation" stage typically begin with only 2–5 posts. Because HDBSCAN requires `min_cluster_size=7`, these micro-signals are classified as noise ($cluster\_id = -1$) and discarded.
* **Diagnosis**: Fixed static thresholds fail to adapt to early-stage signal dynamics.
* **Proposed Fix for Step 9**: Implement two-tier micro-clustering: identify dense 3-post seed clusters using local $k$-NN density before applying full cluster aggregation.

---

### 5. Baseline Mismatch: The False-Alarm Burden
* **Case Study**: 464 Unassociated Baseline Alerts (`free`, `humans`, `week`, `latest`, `camp`)
* **The Error**: The keyword-frequency baseline fired 472 total alerts across the 6 checkpoints. Only 8 alerts corresponded to our vetted events (`onlyfans`, `october`, `lock`, `artists`). The remaining 464 alerts were individual word fluctuations.
* **Diagnosis**: Single-word frequencies in small samples (~1,000 posts/window) have high stochastic variance. A jump from 1 mention to 8 mentions of the word `"humans"` produces an $8.0\times$ surge ratio, triggering a false-alarm alert.
* **Value of the Semantic Detector**: The semantic NLP detector solved this problem by grouping words into dense semantic contexts, reducing the alert burden by $98.3\%$ relative to the 472 baseline triggers.

---

## Actionable Roadmap for Step 9 (Final Experiments & Ablations)

The zero lead-time finding ($0$ checkpoints) and the error taxonomy directly inform our experimental design for Step 9:

1. **Continuous Daily Temporal Stream (Solving the Incubation Blindspot)**:
   * Transition from 6 discrete bi-weekly snapshots to daily rolling slices ($T_1, T_2, \dots, T_30$).
   * Test whether semantic density begins accumulating 48–72 hours before keyword volume surges.
2. **Template Stripping & Trope Classification**:
   * Pre-filter conversational prompt interrogatives to eliminate false emergence (`TRAJ_004`).
3. **Lexical Consistency Gating**:
   * Combine centroid cosine similarity with c-TF-IDF keyword overlap to eliminate semantic drift (`TRAJ_016`).
4. **Adaptive Density Burst Detection**:
   * Replace rigid post count cutoffs with entity-weighted z-score density to recover fragmented public discourse (`TRAJ_017` / `TRAJ_027`).

---
*Generated as part of Step 8: Error Analysis for EmergNLP.*
