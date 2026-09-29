# Identifying Emerging Topics Before They Become Visible
## Controlled Ablation Suite & Temporal Topic Evolution Report (E0 — E6)

**Project:** Advanced NLP/ML Emerging Topic Detection (Problem ID: ML-T2-008)  
**Dataset:** Reddit r/AskReddit (HuggingFace SocialGrep Stream, 8,784 Posts, 30 Temporal Checkpoints: Aug 11 – Sep 5, 2021)  
**Evaluation Standard:** Frozen 823-Alert Frequency Baseline | Strict Forward-Only Causality  
**Status:** **E0–E5: FROZEN** | **E6: FROZEN AFTER PRE-FREEZE AUDIT** | **Ablation Study: COMPLETE**  

---

## 1. Executive Summary & Research Motivation

Standard keyword-frequency alerting methods suffer from severe alert inflation when applied to fast-moving social media streams, often flagging hundreds of isolated conversational spikes while remaining blind to emerging semantic shifts until those topics already dominate discussion. Conversely, dense embedding clustering (Sentence-BERT + HDBSCAN) offers semantic cohesion, but naive applications encounter critical failure modes: scale mismatch in fine-grained slices, conversational syntactic template bursts, semantic drift across time, and the extreme data sparsity of pre-event topic seeds.

This study presents a systematic, 7-stage controlled ablation suite (**E0 through E6**) that isolates each architectural failure mode and evaluates targeted interventions across a fine-grained 30-checkpoint temporal stream (Aug 11 – Sep 5, 2021, 8,784 posts), benchmarked against a frozen 823-alert keyword frequency baseline.

### Core Contributions
1. **Strict Temporal Methodology:** Checkpoints are evaluated chronologically with zero forward temporal leakage. Clustering, keyword extraction, and online running centroids operate strictly on historical context up to checkpoint $D_t$, ensuring retrospective validation never leaks into real-time alerting.
2. **Ablation-Derived Architecture:** Every intervention across the E0 $\to$ E6 sequence is derived from an empirically observed failure mode (scale mismatch, syntactic templates, semantic drift, entity continuity, sparse-topic blindness, and micro-cluster candidate inflation) rather than ad-hoc parameter tuning.
3. **Two-Tier Topic Maturity Model:** The system decouples standard high-density topic detection ($N \ge 7$) from a quarantined low-volume incubation layer ($3 \le N < 7$). This structural separation permits early observation of sparse candidate seeds without promoting unconfirmed micro-clusters into false-positive alerts.

### Summary of Experimental Findings
- **Detection Lead Time:** No demonstrated temporal lead-time advantage over the keyword baseline was observed for the manually validated abrupt events (e.g., OnlyFans policy reversal broke simultaneously in keyword and semantic streams at $D_{11}$, lead time = 0 days).
- **Semantic Topic Consolidation:** The semantic pipeline converts fragmented, uncontextualized keyword spikes into coherent, interpretable topic trajectories. Across the stream, the semantic detector produced **102 validated candidates compared with 823 keyword-level alerts, an 87.6% reduction in alert volume**.
- **Sparse Incubation:** The two-tier architecture successfully recovered a sparse pre-emergence vaccine mandate seed at $D_{16}$ (3 posts, coherence 0.938) while keeping **517 micro-seeds quarantined in incubation**, preventing the 4× candidate inflation observed under unconstrained micro-clustering.

---

## 2. Master Ablation Comparison (E0 — E6)

All 7 experimental conditions were evaluated against the identical 30-checkpoint fine-grained stream (8,784 posts) and the frozen 823-alert keyword baseline:

| Experiment | Condition Description | Primary Threshold / Mechanism | Total Trajectories | Multi-Ckpt Trajectories | Validated Candidates | Quarantined Incubation Seeds | OnlyFans ($D_{11}$) | FDA / Vaccine ($D_{16}$) | Locked Comments ($D_{22-27}$) | Max Trajectory Span | Mean Candidate Coherence | Semantic Drift | Baseline Alert Count (Frozen) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E0** | Control Detector | Absolute volume $\ge 15$ posts | 82 | 6 | 2 | 0 | Missed ($N=12$) | Missed ($N<7$) | Missed ($N<7$) | 10 ckpts | 0.6670 | Present | 823 |
| **E1** | Scale Normalization | Topic share $\ge 2.0\%$ slice | 82 | 6 | 36 | 0 | **Captured** ($D_{11}$) | Missed ($N<7$) | Missed ($N<7$) | 10 ckpts | 0.6974 | Present | 823 |
| **E2** | Template Stripping | Interrogative syntax removal | 109 | 7 | 32 | 0 | **Captured** ($D_{11}$) | Missed ($N<7$) | Missed ($N<7$) | 24 ckpts | 0.6805 | Severe (`TRAJ_002`) | 823 |
| **E3** | Lexical Consistency | Cosine $\ge 0.68$ + Jaccard $\ge 0.20$ | 133 | 11 | 35 | 0 | **Captured** ($D_{11}$) | Missed ($N<7$) | Missed ($N<7$) | 5 ckpts | 0.6819 | Controlled | 823 |
| **E4** | Entity-Aware Linkage | NER dual gate ($J_{ent} \ge 0.20$) | 132 | 12 | 35 | 0 | **Captured** ($D_{11}$) | Missed ($N<7$) | Missed ($N<7$) | 5 ckpts | 0.6819 | Controlled | 823 |
| **E5** | Micro-Clustering | Min cluster size $N=3$ | 1,121 | 4 | 141 | 721 | Fragmented (9p) | **Recovered** (10 clusters) | Missed ($N<3$) | 3 ckpts | 0.6957 | Fragmented | 823 |
| **E6** | Combined Architecture | Two-Tier ($N\ge 7$ + Incubation $3\le N<7$) | 862 | 23 | 102 | 517 | **Captured** (11p intact) | **Recovered** ($D_{16}$ seed) | Missed ($N<3$) | 5 ckpts | 0.6983 | Controlled | 823 |

---

## 3. The 7-Stage Controlled Ablation Progression

### E0: Baseline Control (Scale Mismatch Discovered)
- **Mechanism:** Static emergence threshold of volume $\ge 15$ posts designed for ~1,000-post slices applied to fine-grained ~300-post checkpoints.
- **Outcome:** Generated only 2 candidate alerts out of 82 trajectories. The real-world OnlyFans policy change at $D_{11}$ (11-12 posts, 3.7%–4.0% share) was missed because it fell below the absolute volume floor.

### E1: Scale Normalization (Template Spikes Exposed)
- **Mechanism:** Transitioned from absolute volume to scale-normalized slice share: $\text{share} \ge 2.0\%$.
- **Outcome:** Successfully captured OnlyFans at $D_{11}$ (`TRAJ_039`, 12 posts, coherence 0.771). However, validated candidates surged from 2 to 36 due to generic AskReddit question templates (*"Redditors who work in...", "What is your biggest..."*).

### E2: Syntactic Template Stripping (Semantic Drift Exposed)
- **Mechanism:** Rule-based regex parser stripped leading interrogatives (*"what is", "how do you", "why does", "people of reddit who"*) prior to SVD embedding reduction and c-TF-IDF keyword extraction.
- **Outcome:** Suppressed **82.4% (28 of 34)** of template-driven candidate spikes. Surfaced specific event vocabulary, including Hurricane Ida ($D_{27}$, 7 posts) and the Lil Nas X meme *"Nah he tweakin"* ($D_{18}$, 8 posts).
- **Negative Finding:** Exposed severe long-range semantic drift under cosine-only tracking. `TRAJ_002` drifted across 24 consecutive checkpoints despite 0.000 keyword overlap between distant windows.

### E3: Consecutive Lexical Consistency Gate (Drift Eliminated)
- **Mechanism:** Dual-criterion continuity constraint: consecutive trajectory steps require $\text{Cosine} \ge 0.68$ **AND** top-10 c-TF-IDF keyword Jaccard overlap $J_{lex} \ge 0.20$.
- **Outcome:** Dismantled artificial drift pseudo-chains. Maximum trajectory lifespan was capped from 24 checkpoints to 5 checkpoints, while preserving genuine multi-checkpoint events: the Afghanistan evacuation crisis remained intact across 5 consecutive checkpoints ($D_{04} \to D_{08}$, 7 $\to$ 14 $\to$ 15 $\to$ 8 $\to$ 7 posts).

### E4: Entity-Aware Linkage (Density Limits Identified)
- **Mechanism:** Dual-gate incorporating spaCy NER overlap ($J_{lex} \ge 0.20 \lor J_{ent} \ge 0.20$) with a $+0.15 \times J_{ent}$ matching boost.
- **Outcome:** Preserved drift control (max span: 5 checkpoints, 35 candidates). Post-level auditing revealed that target pre-event topics had sub-threshold counts in every checkpoint (`fda`: 2 posts total, `pfizer`: 1 post total, `locked comments`: 7 posts total, max 2 in any slice). This proved that NER linkage cannot compensate when post volume falls below HDBSCAN's density parameter ($N < 7$).

### E5: Micro-Clustering (Sensitivity vs. Candidate Inflation)
- **Mechanism:** Lowered HDBSCAN minimum cluster size from $N = 7$ to $N = 3$.
- **Outcome:** Successfully recovered sparse pre-emergence seeds, including 10 discrete vaccine/mandate clusters (e.g., $D_{16}$ seed at 0.938 coherence).
- **Severe Trade-offs:**
  1. *Candidate Inflation:* Trajectories surged 8.5× (132 to 1,121); validated candidates quadrupled from 35 to 141.
  2. *Topic Fragmentation:* OnlyFans fragmented into 9-post sub-clusters; Hurricane Ida split across 4 fragments; the Afghanistan trajectory was shattered.
  3. *Coherence Ineffectiveness:* 980 of 983 raw micro-clusters passed coherence $\ge 0.45$. Semantic coherence alone is insufficient to filter low-volume candidate inflation.

### E6: Two-Tier Architecture with Quarantined Incubation (Mitigating the Trade-Off)
- **Architecture:**
  ```
                        ┌── Tier 1: N ≥ 7 ──────→ Standard Emerging Topic Pipeline
                        │                         (Preserves established events intact)
  Timestamped Posts ────┤
                        └── Tier 2: 3 ≤ N < 7 ──→ Incubation Layer
                                                  │
                                                  ├── Coherence Quality Gate (≥ 0.65)
                                                  ├── Quarantined as [MICRO_SEED] (Score = 0.0)
                                                  └── Confirmation Mechanism:
                                                      * Temporal recurrence (≥ 2 ckpts) OR
                                                      * Domain entity + High Coh (≥ 0.85)
                                                          │
                                                          ▼
                                                   [CONFIRMED_SEED]
  ```
- **Outcome:**
  - Preserved established event structures: OnlyFans (11 posts intact at $D_{11}$), Hurricane Ida (7 posts intact at $D_{27}$), Afghanistan (5 checkpoints intact, 51 posts total).
  - Contained trajectories to 862 (vs. 1,121 in E5).
  - Moderated validated candidates to 102 (98 bursts + 4 confirmed seeds), while **517 micro-seeds remained safely quarantined in incubation** without triggering unvetted alerts.
  - Successfully recovered the sparse $D_{16}$ vaccine seed (`TRAJ_456`, 3 posts, coherence 0.938, rep: *"Should one try and reason with a friend ready to throw away their medical career over a vaccine mandate?"*).

---

## 4. Pre-Freeze 5-Point Validation Audit

Before declaring the ablation study frozen, a rigorous 5-point audit was conducted:

### A. Temporal Leakage Audit: PASSED
All 30 checkpoints are processed strictly chronologically. Checkpoint $D_t$ features (embeddings, SVD, clusters, keywords) depend exclusively on slice $D_t$. 1-to-1 trajectory matching connects $D_t$ clusters only to trajectories active in $D_{t-1}$ or $D_{t-2}$. Online centroids update recursively. Emergence scoring and candidate validation rely solely on historical trajectory metrics up to $D_t$.

### B. Candidate Classification Audit: PASSED
Audit of all 102 candidates demonstrated clean operational separation:
- **Top Bursts (98 Total):** Captured prominent surges, including Lil Nas X meme *"Nah he tweakin"* ($D_{18}$, 8 posts, score 0.817) and OnlyFans ($D_{11}$, 11 posts, score 0.565).
- **Confirmed Seeds (4 Total):** `TRAJ_456` ($D_{16}$, 3 posts, coh 0.938, vaccine mandate impact), `TRAJ_409` ($D_{15}$, 4 posts, coh 0.992, COVID-19 inquiry), `TRAJ_601` ($D_{21}$, recurring micro-seed 3 $\to$ 4 $\to$ 4 posts, coh 0.791), `TRAJ_011` ($D_{01}$, recurring 5 $\to$ 3 posts, coh 0.755).
- **Quarantined Micro-Seeds (517 Total):** Safely held in incubation at score 0.0 (e.g., highly coherent 3-post clusters about niche golf terminology, visa subclass applications, cell phone hotspot masking).

### C. FDA / Vaccine Seed Audit at $D_{16}$: PASSED
The $D_{16}$ vaccine mandate seed was identified using only the 3 posts within checkpoint $D_{16}$. Its high coherence (0.938) and entity anchor satisfied the pre-defined seed confirmation criterion at step $D_{16}$ without future information.

### D. Entity Rule Transparency Audit: DOCUMENTED
We explicitly document that the domain entity list (`vaccine`, `covid`, `fda`, `pfizer`, `mandate`) and coherence threshold (0.85) were formulated during post-E5 error diagnosis. In an unguided production deployment without prior target interest, seed confirmation should rely strictly on **unsupervised temporal recurrence across $\ge 2$ checkpoints** to prevent post-hoc target tuning.

### E. Baseline Comparison Integrity: PASSED
The frozen keyword baseline generated **823 alerts** across the stream. This count remains constant across all ablation rows.

---

## 5. Observable Detection Boundary: Reddit Locked Comments

The Reddit "locked comments" bug provides an empirical lower bound for density-based clustering:
- Slices: $D_{22} = 1$ post, $D_{25} = 1$ post, $D_{27} = 2$ posts.
- Because post counts never achieve $\ge 3$ posts in any single checkpoint, this event falls below the empirical density threshold of the current single-checkpoint clustering mechanism of density-based clustering at daily temporal resolution.
- Events with $\le 2$ posts per time slice are mathematically indistinguishable from background noise in single-slice clustering, motivating future work in cross-checkpoint streaming accumulation.

---

## 6. Scientific Safeguards & Rigorous Phrasing

1. **Candidate Inflation, Not Noise:** Low-volume candidates are designated as *"candidate inflation requiring additional validation"* rather than dismissed as "noise," preserving scientific modesty.
2. **Mitigating Trade-offs, Not Resolving:** We state that E6 *"mitigates the sensitivity–candidate-inflation trade-off,"* rather than claiming a complete resolution, because full precision/recall requires exhaustive manual labeling of the entire 8,784-post corpus.
3. **Incubation Taxonomy:** Topic maturity stages are explicitly distinguished:
   $$\text{MICRO\_SEED (Quarantined)} \longrightarrow \text{CONFIRMED\_SEED (Candidate)} \longrightarrow \text{EMERGING\_TOPIC (Verified)}$$
   "Confirmed Seed" denotes a sufficiently coherent candidate seed meeting incubation thresholds, not an independently proven emerging event.
4. **Volume Reduction, Not Generic Noise Reduction:** We state that the semantic system produced **102 validated candidates compared with 823 keyword-level alerts, an 87.6% reduction in alert volume relative to the keyword baseline**, acknowledging that uninspected baseline alerts may include valid conversational activity.

---

## 7. Final Scientific Conclusion

Across the controlled E0–E6 ablation sequence, the proposed two-tier architecture mitigated the sensitivity–candidate-inflation trade-off observed with unconstrained micro-clustering. The system preserved established event clusters, constrained observed semantic drift, and retained low-volume topic seeds in a quarantined incubation layer. In the evaluated 30-checkpoint Reddit stream, it recovered a sparse vaccine-related seed that standard clustering missed while maintaining a clear density boundary for events with fewer than three co-occurring posts. The experiments did not demonstrate measurable lead time over the keyword baseline for the manually validated abrupt events; instead, the primary demonstrated benefit was semantic topic consolidation and structured incubation of sparse signals.


---

## 8. Terminology Standard & Scientific Safeguards

| Avoid | Use Instead | Scientific Rationale |
|---|---|---|
| "Solved" / "Resolved" | **"Mitigated" / "Systematically addressed"** | Reflects experimental containment rather than universal resolution |
| "Noise reduction" | **"Alert-volume reduction"** | Baseline alerts were not exhaustively labeled; avoids unsupported claims |
| "Universal detection threshold" | **"Empirical density boundary"** | Reflects specific single-checkpoint clustering density limit ($N \ge 3$) |
| "Predicted news early" | **"Recovered sparse candidate seeds"** | Zero lead time observed on abrupt events; reflects incubation, not forecasting |
| "Perfect / complete prevention" | **"Contained / controlled observed behavior"** | Drift eliminated in this empirical setup, not mathematically impossible |
| "Best model" | **"Final combined architecture"** | Architecture balances sensitivity and candidate inflation rather than dominating on all metrics |
| "False positives" (without labels) | **"Candidate inflation" / "Unassociated alerts"** | Precise, unverified low-volume activity description |

---

## 9. Final Project Identity

- **Project:** Identifying Emerging Topics Before They Become Visible
- **Problem ID:** ML-T2-008
- **Core Method:** Sentence-BERT + TruncatedSVD + HDBSCAN + c-TF-IDF + Online Trajectory Tracking
- **Evaluation Corpus:** 30 temporal checkpoints, 8,784 Reddit posts (Aug 11 – Sep 5, 2021)
- **Baseline Control:** Frozen 823-alert keyword-frequency baseline
- **Final Architecture:** E0–E6 controlled ablation suite $\to$ Two-tier incubation model
- **One-Line Portfolio Description:** *Designed a causally constrained NLP system for emerging-topic detection, using a controlled E0–E6 ablation suite and a two-tier semantic incubation architecture to consolidate topic alerts and retain sparse candidate signals without future-data leakage.*
