# EmergNLP: AI-Powered Emerging Topic Detection & Intelligence System

> **Executive Summary:** EmergNLP is a causally constrained NLP system designed to monitor fast-moving social media streams (Reddit, Twitter/X, forums) to detect emerging trends, breaking news, and organic community discussions before they peak—**reducing false-alarm alert noise by 87.6%** compared to traditional keyword-frequency monitoring.

---

## Table of Contents
1. [Business Problem & The "Alert Fatigue" Challenge](#1-business-problem--the-alert-fatigue-challenge)
2. [How EmergNLP Works (System Architecture)](#2-how-emergnlp-works-system-architecture)
3. [Step-by-Step Technical Workflow](#3-step-by-step-technical-workflow)
4. [Empirical Validation & Benchmark Results](#4-empirical-validation--benchmark-results)
5. [The Controlled Ablation Progression (E0 to E6)](#5-the-controlled-ablation-progression-e0-to-e6)
6. [Future Roadmap & Strategic Goals](#6-future-roadmap--strategic-goals)
7. [Quickstart: How to Run the Dashboard](#7-quickstart-how-to-run-the-dashboard)
8. [Repository Architecture](#8-repository-architecture)

---

## 1. Business Problem & The "Alert Fatigue" Challenge

In modern brand intelligence, crisis monitoring, and financial surveillance, organizations track high-volume social media streams to catch breaking events early.

Traditional monitoring tools rely on **keyword frequency tracking** (e.g., alert when a word spikes $>3\sigma$ above average). This causes severe operational failure:
* **Alert Fatigue:** Common words like *"help"*, *"school"*, *"money"*, or *"people"* spike repeatedly without representing actual news events. In our 30-day benchmark, keyword monitoring produced **823 disconnected alert spikes**.
* **Zero Semantic Context:** A keyword alert like `"ban"` gives an analyst no indication of what is being banned or why.
* **Semantic Fragmentation:** Ten different synonyms discussing the same crisis trigger ten isolated alerts rather than a single unified event.

**EmergNLP solves this** by grouping discussions by semantic meaning using dense vector representations, tracking topic evolution over time, and filtering syntactic platform noise.

---

## 2. How EmergNLP Works (System Architecture)

```
                            RAW SOCIAL MEDIA POST STREAM
               (8,784 Posts across 30 Chronological Checkpoints, Zero Future Leakage)
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ STEP 1: Preprocessing & Syntactic Template Stripping    │
            │ (Regex parser eliminates boilerplate questions)        │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ STEP 2: Dense Semantic Encoding (Sentence-BERT)        │
            │ (384-dimensional dense vectors via all-MiniLM-L6-v2)   │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ STEP 3: Unsupervised Density Clustering (HDBSCAN)      │
            │ (Groups posts into coherent topics + c-TF-IDF keywords)│
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ STEP 4: Causally Constrained Trajectory Tracking       │
            │ (Cosine Sim ≥ 0.68 + Keyword Jaccard Overlap ≥ 0.20)   │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ STEP 5: Two-Tier Emergence Scoring & Incubation        │
            └──────────────┬──────────────────────────┬──────────────┘
                           │                          │
        Tier 1 (High Volume: N ≥ 7)      Tier 2 (Sparse Seed: 3 ≤ N < 7)
                           │                          │
                           ▼                          ▼
              [EMERGING TOPIC ALERT]         [INCUBATION QUARANTINE]
              • Scale-normalized share       • Monitored across days
              • Acceleration velocity        • High coherence gate (≥0.65)
              • Actionable event alert       • Confirmed only if recurrent
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ STEP 6: Interactive Intelligence Dashboard (Streamlit) │
            │ (Real-time inspection, lead-time audit, trajectory map)│
            └────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Technical Workflow

### 1. Ingestion & Strict Forward-Only Causality (`src/data_loader.py`)
Data is ingested into chronological time slices ($D_{00}$ to $D_{29}$). Checkpoints are processed sequentially with zero lookahead, guaranteeing that retrospective evaluation never leaks future information into real-time alerting.

### 2. Syntactic Template Stripping (`src/preprocessing.py`)
Social platforms like Reddit have dominant question templates (*"Redditors who work in...", "What is the best..."*). Naive NLP systems group these together because they share grammatical structure. EmergNLP's preprocessor strips leading interrogatives and boilerplate syntax before embedding, forcing clusters to form around **topical substance**.

### 3. Dense Semantic Representation (`src/embeddings.py`)
Each cleaned post is mapped into continuous vector space using Sentence-BERT (`all-MiniLM-L6-v2`). Posts discussing *"vaccine mandate"* and *"immunization requirement"* map to adjacent points in vector space even if they share zero identical keywords.

### 4. Density-Based Clustering (`src/clustering.py`)
HDBSCAN groups vectors into semantic clusters without requiring the user to specify the number of topics in advance. Class-based TF-IDF (c-TF-IDF) extracts the most salient keywords that uniquely identify each cluster.

### 5. Trajectory Tracking & Drift Containment (`src/temporal.py`)
Clusters are connected across consecutive days into long-term trajectories. To eliminate **semantic drift** (topics slowly morphing over weeks while falsely remaining linked), EmergNLP enforces a strict dual gate:
$$\text{Centroid Cosine Similarity} \ge 0.68 \quad\text{AND}\quad \text{Keyword Jaccard Overlap} \ge 0.20$$

### 6. Two-Tier Emergence & Incubation (`src/emergence.py`)
* **Scale-Normalized Emergence:** Rather than relying on raw volume thresholds, the system flags emergence when a topic's slice volume share exceeds $2.0\%$ and shows positive acceleration.
* **Quarantined Incubation Layer:** Sparse micro-clusters ($3 \le N < 7$ posts) are placed in a quarantined holding room (`MICRO_SEED`, score = 0.0). They are only upgraded to validated alerts (`CONFIRMED_SEED`) if they recur across time slices or meet extreme coherence criteria ($\ge 0.85$), preventing false alarms.

---

## 4. Empirical Validation & Benchmark Results

The system was benchmarked against a frozen 823-alert keyword frequency baseline across 8,784 posts:

| Metric | Keyword Frequency Baseline | EmergNLP Pipeline (E6) | Business Impact |
|---|---|---|---|
| **Total Alerts Generated** | **823 alerts** | **102 candidate topics** | **87.6% reduction in alert noise** |
| **Topic Context** | Isolated keywords (`"ban"`, `"school"`) | Full topic summaries + representative posts | Analysts read full event narratives instantly |
| **Real Events Identified** | Diluted inside 800+ false alarms | Cleanly surfaced OnlyFans policy shift, Hurricane Ida, Vaccine Mandates | Zero alert fatigue for intelligence teams |
| **Pre-Event Seed Recovery** | Completely invisible | Successfully recovered 3-post vaccine seed ($D_{16}$, Coherence: 0.938) | Captures early discussion before peak surge |
| **Quarantined Noise** | N/A (Spammed to user) | 517 micro-seeds held safely in incubation | Prevents candidate inflation |

---

## 5. The Controlled Ablation Progression (E0 to E6)

Every component in EmergNLP was systematically validated through an ablation study:

* **E0 (Baseline Control):** Proved static volume floors ($\ge 15$ posts) miss critical breaking news (e.g., OnlyFans at 12 posts was missed).
* **E1 (Scale Normalization):** Using topic share ($\ge 2.0\%$) successfully captured OnlyFans, but exposed conversational question templates.
* **E2 (Template Stripping):** Suppressed **82.4%** of template noise and surfaced Hurricane Ida and viral events, but exposed long-term semantic drift.
* **E3 (Lexical Consistency Gate):** Eliminated artificial 24-day drift chains, capping trajectory lifespan to authentic events (e.g., the 5-day Afghanistan evacuation crisis).
* **E4 (Entity-Aware Linkage):** Added NER dual gating; identified that topics with $<3$ posts require structural incubation.
* **E5 (Micro-Clustering $N=3$):** Unlocked sensitivity to small seeds, but caused 4× candidate inflation and topic fragmentation.
* **E6 (Combined Two-Tier Architecture):** **Final Production Architecture.** Decoupled standard detection ($N \ge 7$) from quarantined incubation ($3 \le N < 7$), achieving maximum sensitivity with minimal clutter.

---

## 6. Future Roadmap & Strategic Goals

### Phase 1: Algorithmic & Research Extensions
1. **Cross-Checkpoint Streaming Accumulation (Sub-3 Post Density Limit):**
   * *Current Boundary:* Events with only 1–2 posts in a single day (e.g., the Reddit locked comments bug) fall below single-checkpoint clustering density ($N < 3$).
   * *Goal:* Implement a **48–72 hour continuous sliding vector buffer** to aggregate sparse signals across days before clustering.
2. **True Temporal Lead-Time Benchmark:**
   * Evaluate the model across organic incubation domains (e.g., HackerNews, Twitter/X, biomedical preprints, financial discussion boards) where trends simmer over days before exploding.
3. **Exhaustive Ground-Truth Annotation:**
   * Expand manually validated event labels across the entire corpus for comprehensive precision/recall/F1 scoring.

### Phase 2: Enterprise Engineering & Production Deployment
1. **Production REST & WebSocket API (FastAPI):**
   * Develop a low-latency API exposing `/alerts/live`, `/topics/active`, and `/incubation/candidates`.
2. **Real-Time Streaming Ingestion:**
   * Connect live streaming consumers (Reddit Streaming API via PRAW, Kafka, or AWS Kinesis) paired with an incremental vector database (**Qdrant**, **Milvus**, or **FAISS**).
3. **Automated Alert Dispatch Webhooks:**
   * One-click integrations with **Slack**, **Discord**, **PagerDuty**, and **Microsoft Teams** for instant notification when high-velocity topics emerge.
4. **Multilingual & Cross-Platform Fusion:**
   * Deploy multilingual Sentence-BERT models to correlate topic emergence simultaneously across Reddit, Twitter/X, news feeds, and customer support tickets.

---

## 7. Quickstart: How to Run the Dashboard

### Prerequisites
* Python 3.9+
* Required packages listed in `requirements.txt`

### Installation & Execution
```bash
# 1. Clone the repository and enter the directory
cd EmergNLP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive dashboard
python run.py
```

### Navigating the Dashboard
Once the browser opens at `http://localhost:8501`:
* **Sections 1–3:** Inspect raw Reddit posts, discovered trajectories, and emerging alerts.
* **Sections 4–5:** Review the comparative analysis vs. the 823-alert keyword baseline.
* **Section 6:** Interactive Topic Inspector (search topics, inspect posts, view keyword distributions).
* **Section 7:** Research Limitations & Observable Density Boundary.
* **Section 8:** Full Controlled Ablation Suite (interactively explore E0 through E6).

---

## 8. Repository Architecture

```
EmergNLP/
│
├── app/
│   └── streamlit_app.py           # Interactive web dashboard & evaluation visualizer
│
├── src/                           # Core Algorithmic & NLP Backend
│   ├── data_loader.py             # Data ingestion and temporal slicing
│   ├── preprocessing.py          # Syntactic template stripping & text cleaning
│   ├── embeddings.py              # Sentence-BERT dense vector generation
│   ├── clustering.py              # HDBSCAN density clustering & c-TF-IDF
│   ├── temporal.py                # Trajectory tracking & lexical consistency gates
│   ├── emergence.py               # Scale-normalized scoring & two-tier incubation
│   ├── baseline.py                # Keyword frequency z-score baseline
│   ├── evaluation.py              # Lead-time & metric evaluation engine
│   └── run_e0_daily.py            # Automated ablation suite execution runner
│
├── data/                          # Processed datasets, trajectories & embeddings
├── experiments/                   # Outputs and metrics for ablation suite E0 through E6
├── reports/                       # Formal research paper (ablation_study_report.md)
├── tests/                         # Automated unit and regression test suite
└── requirements.txt               # Project Python dependencies
```
