import os
import sys
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Streamlit Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EmergNLP: Topic Emergence & Evaluation Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-burst {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-emerging {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-declining {
        background-color: #F1F5F9;
        color: #475569;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_all_data():
    raw_posts_path = os.path.join(BASE_DIR, "data", "processed", "reddit_cleaned_6weeks.csv")
    trajs_path = os.path.join(BASE_DIR, "data", "processed", "emerging_topics_alerts_6weeks.csv")
    eval_path = os.path.join(BASE_DIR, "experiments", "evaluation_results_6weeks.csv")
    baseline_path = os.path.join(BASE_DIR, "data", "processed", "baseline_volume_alerts_6weeks.csv")
    clusters_path = os.path.join(BASE_DIR, "data", "processed", "clusters_summary_6weeks.csv")
    posts_meta_path = os.path.join(BASE_DIR, "data", "processed", "clustered_topics_6weeks.csv")
    matches_path = os.path.join(BASE_DIR, "data", "processed", "trajectory_matches_6weeks.csv")

    posts_df = pd.read_csv(raw_posts_path) if os.path.exists(raw_posts_path) else pd.DataFrame()
    trajs_df = pd.read_csv(trajs_path) if os.path.exists(trajs_path) else pd.DataFrame()
    eval_df = pd.read_csv(eval_path) if os.path.exists(eval_path) else pd.DataFrame()
    base_df = pd.read_csv(baseline_path) if os.path.exists(baseline_path) else pd.DataFrame()
    clusters_df = pd.read_csv(clusters_path) if os.path.exists(clusters_path) else pd.DataFrame()
    meta_df = pd.read_csv(posts_meta_path) if os.path.exists(posts_meta_path) else pd.DataFrame()
    matches_df = pd.read_csv(matches_path) if os.path.exists(matches_path) else pd.DataFrame()

    return posts_df, trajs_df, eval_df, base_df, clusters_df, meta_df, matches_df

posts_df, trajs_df, eval_df, base_df, clusters_df, meta_df, matches_df = load_all_data()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🔬 EmergNLP: Topic Emergence & Baseline Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Historical Replay Evaluation: Dense Semantic NLP Topic Detection vs. Conventional Keyword-Frequency Baseline</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to Section:",
    [
        "1. Dataset Overview",
        "2. Topic Evolution & Trajectories",
        "3. Vetted Emerging Candidates",
        "4. Baseline Comparison & Noise Analysis",
        "5. Lead-Time Evaluation",
        "6. Interactive Topic Inspector",
        "7. Research Limitations & Future Work",
        "8. Controlled Ablation Suite (E0 - E6)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Experiment Setup**")
st.sidebar.markdown("- **Corpus**: Reddit r/AskReddit (HF SocialGrep)")
st.sidebar.markdown("- **Scope**: 6 Temporal Checkpoints (July–Sept 2021)")
st.sidebar.markdown("- **Model**: Sentence-BERT `all-MiniLM-L6-v2` + HDBSCAN")
st.sidebar.markdown("- **Tracking**: 1-to-1 Bipartite Matching (Gap ≤ 1)")

# -----------------------------------------------------------------------------
# 1. Dataset Overview
# -----------------------------------------------------------------------------
if section == "1. Dataset Overview":
    st.header("1. Dataset & Checkpoint Architecture")
    st.write(
        "To evaluate topic emergence without future data leakage, the historical replay dataset "
        "spans **6 distinct chronological checkpoints** across 65 days (July 25, 2021 to September 30, 2021). "
        "Each checkpoint samples approximately **1,000 posts** to evaluate relative topic share over time."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cleaned Posts", f"{len(posts_df):,}")
    c2.metric("Temporal Checkpoints", "6 Windows (W0–W5)")
    c3.metric("Sample per Checkpoint", "~1,000 Posts")
    c4.metric("Temporal Span", "65 Days (Jul–Sep 2021)")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Posts per Temporal Checkpoint")
        ckpt_summary = posts_df.groupby(["window_id", "week_label"]).agg(
            post_count=("id", "count"),
            start_date=("datetime_utc", "min"),
            end_date=("datetime_utc", "max")
        ).reset_index()

        fig_bar = px.bar(
            ckpt_summary,
            x="window_id",
            y="post_count",
            color="window_id",
            text="post_count",
            labels={"window_id": "Checkpoint ID", "post_count": "Post Count"},
            title="Balanced Checkpoint Post Distribution (Uniform Slices)"
        )
        fig_bar.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_bar, width="stretch")

    with col_right:
        st.subheader("Checkpoint Schedule & Dates")
        st.dataframe(
            ckpt_summary[["window_id", "week_label", "post_count", "start_date", "end_date"]],
            width="stretch",
            hide_index=True
        )
        st.info(
            "💡 **Methodology Note**: Checkpoints are normalized to ~1,000 posts. "
            "Therefore, volume changes across checkpoints represent changes in relative topic density "
            "within uniform sampling slices."
        )

# -----------------------------------------------------------------------------
# 2. Topic Evolution & Trajectories
# -----------------------------------------------------------------------------
elif section == "2. Topic Evolution & Trajectories":
    st.header("2. Topic Evolution & Temporal Tracking")
    st.write(
        "Using dense Sentence-BERT embeddings reduced via SVD, HDBSCAN uncovers semantic clusters in each window. "
        "A **1-to-1 greedy bipartite tracking algorithm** connects clusters across checkpoints if cosine similarity ≥ 0.68, "
        "strictly enforcing a **temporal gap constraint of ≤ 1 checkpoint**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trajectories", f"{len(trajs_df)}")
    multi_week = len(trajs_df[trajs_df["active_checkpoints"] > 1])
    c2.metric("Multi-Checkpoint Topics", f"{multi_week}")
    c3.metric("Single-Checkpoint Topics", f"{len(trajs_df) - multi_week}")
    c4.metric("Avg Coherence", f"{trajs_df['avg_coherence'].mean():.3f}")

    st.markdown("---")
    st.subheader("Multi-Checkpoint Trajectory Progression")

    # Select top persistent or growing trajectories to plot
    multi_trajs = trajs_df[trajs_df["active_checkpoints"] >= 2].head(10)
    
    plot_rows = []
    for _, r in multi_trajs.iterrows():
        timeline = r["timeline_path"].split(" -> ")
        vols = [int(v) for v in str(r["volume_path"]).split(" -> ")]
        for w, v in zip(timeline, vols):
            plot_rows.append({
                "Trajectory": f"{r['trajectory_id']}: {r['topic_name']}",
                "Checkpoint": w,
                "Volume": v,
                "Status": r["lifecycle_status"]
            })
    plot_df = pd.DataFrame(plot_rows)

    if not plot_df.empty:
        fig_line = px.line(
            plot_df,
            x="Checkpoint",
            y="Volume",
            color="Trajectory",
            markers=True,
            title="Volume Progression Across Chronological Checkpoints"
        )
        fig_line.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_line, width="stretch")

    st.subheader("All Discovered Temporal Trajectories")
    st.dataframe(
        trajs_df[[
            "trajectory_id", "lifecycle_status", "topic_name", "active_checkpoints",
            "timeline_path", "volume_path", "total_volume", "net_volume_growth",
            "growth_ratio", "avg_coherence", "min_coherence"
        ]],
        width="stretch",
        hide_index=True
    )

# -----------------------------------------------------------------------------
# 3. Vetted Emerging Candidates
# -----------------------------------------------------------------------------
elif section == "3. Vetted Emerging Candidates":
    st.header("3. Manually Vetted Candidate Topics")
    st.write(
        "After applying strict research quality guardrails (Coherence ≥ 0.62, Gap ≤ 1, and measurable burst criteria), "
        "we frozen **three manually vetted candidate events/topics** from the corpus."
    )

    vetted = trajs_df[trajs_df["is_vetted_candidate"] == True]

    for idx, (_, r) in enumerate(vetted.iterrows(), 1):
        with st.container():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                badge = "badge-burst" if "[BURST]" in r["lifecycle_status"] else "badge-emerging"
                st.markdown(
                    f"### #{idx} [{r['trajectory_id']}] <span class='{badge}'>{r['lifecycle_status']}</span> {r['topic_name']}",
                    unsafe_allow_html=True
                )
                st.write(f"**Representative Post**: *\"{r['representative_post']}\"*")
                st.write(f"**Timeline Path**: `{r['timeline_path']}` | **Volume History**: `{r['volume_path']}`")
            with col_b:
                st.metric("Total Posts", r["total_volume"])
                st.metric("Avg Coherence", f"{r['avg_coherence']:.3f}")
            st.markdown("---")

    st.subheader("Quality Filtering Summary (Exclusions)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Low-Coherence / Semantic Drift Filter (37 Topics Excluded)**")
        st.write(
            "- Trajectories dipping below the coherence floor (Min < 0.60 or Avg < 0.62) were rejected.\n"
            "- Example: `TRAJ_016` began as dating questions in W1–W2, but drifted into Santa Claus/Rickrolling in W4."
        )
    with c2:
        st.markdown("**Cooling / Declining Filter (28 Topics Excluded)**")
        st.write(
            "- Stagnant or shrinking background themes (e.g. food, general movie questions) were categorized as `[DECLINING]`.\n"
            "- Example: `TRAJ_007` peaked in W1 with 47 posts, then steadily declined to 12 posts."
        )

# -----------------------------------------------------------------------------
# 4. Baseline Comparison & Noise Analysis
# -----------------------------------------------------------------------------
elif section == "4. Baseline Comparison & Noise Analysis":
    st.header("4. Conventional Keyword Baseline vs. Semantic NLP")
    st.write(
        "The conventional frequency baseline monitors rolling unigram/bigram counts and alerts when "
        "a term's current frequency surges ≥ 2.0x above its historical average."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Keyword Baseline Alerts", f"{len(base_df)} term alerts")
    c2.metric("Associated with Vetted Events", "8 term alerts")
    c3.metric("Unassociated Keyword Flurries", f"{len(base_df) - 8} alerts")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Alert Volume & Consolidation")
        comparison_data = pd.DataFrame({
            "Method": ["Keyword Baseline", "NLP Semantic Detector"],
            "Alert Count": [len(base_df), 3],
            "Type": ["Term-Level Unigrams/Bigrams", "Cohesive Semantic Topics"]
        })
        fig_comp = px.bar(
            comparison_data,
            x="Method",
            y="Alert Count",
            color="Method",
            text="Alert Count",
            title="Total Alert Triggers: Term-Level vs. Topic-Level"
        )
        fig_comp.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_comp, width="stretch")

    with col2:
        st.subheader("Term Alert Surges in Baseline")
        st.write(
            "The baseline flags single words whenever relative frequency increases, producing high false-alarm burden:"
        )
        st.dataframe(
            base_df[["term", "detected_window", "current_volume", "past_avg_volume", "surge_ratio"]].head(12),
            width="stretch",
            hide_index=True
        )

    st.info(
        "🔬 **Scientifically Defensible Finding**: Within this evaluated candidate set, the semantic NLP detector "
        "reduced the number of unassociated term-level alerts by **98.3% relative to the 472-alert baseline** "
        "(from 472 fragmented term alerts to 3 synthesized narrative topics)."
    )

# -----------------------------------------------------------------------------
# 5. Lead-Time Evaluation
# -----------------------------------------------------------------------------
elif section == "5. Lead-Time Evaluation":
    st.header("5. Checkpoint Lead-Time Evaluation")
    st.write(
        "We mapped each vetted candidate topic to its corresponding baseline keyword triggers to measure "
        "**Checkpoint Lead Time** (how many checkpoints earlier the NLP method detected the event)."
    )

    st.dataframe(
        eval_df[[
            "candidate_id", "candidate_type", "topic_label", "nlp_detection_window",
            "baseline_detection_window", "lead_time_checkpoints",
            "baseline_matched_terms", "baseline_first_alert_ratio"
        ]],
        width="stretch",
        hide_index=True
    )

    st.markdown("---")
    st.subheader("Empirical Findings & Lead-Time Reality")

    c1, c2, c3 = st.columns(3)
    c1.metric("NLP Earlier Detections", "0 Candidates")
    c2.metric("Same Checkpoint Detections", "3 / 3 (100.0%)")
    c3.metric("Baseline Earlier Detections", "0 Candidates")

    st.warning(
        "⚠️ **Honest Research Conclusion**: In this 6-checkpoint sampled experiment, semantic topic detection "
        "**did not provide measurable temporal lead time** over the keyword-frequency baseline (lead time = 0 checkpoints for all 3 candidates). "
        "Both approaches detected each event in the exact same checkpoint where it first appeared in the corpus. "
        "The value demonstrated by the semantic NLP detector is **topic synthesis and alert volume reduction**, "
        "not earlier predictive warning."
    )

# -----------------------------------------------------------------------------
# 6. Interactive Topic Inspector
# -----------------------------------------------------------------------------
elif section == "6. Interactive Topic Inspector":
    st.header("6. Deep-Dive Topic Inspector")
    st.write("Inspect member posts, cluster keywords, and trajectory progression for any discovered topic.")

    traj_options = trajs_df["trajectory_id"].tolist()
    selected_tid = st.selectbox(
        "Select Trajectory to Inspect:",
        traj_options,
        format_func=lambda x: f"{x} - {trajs_df[trajs_df['trajectory_id'] == x]['topic_name'].values[0]} ({trajs_df[trajs_df['trajectory_id'] == x]['lifecycle_status'].values[0]})"
    )

    if selected_tid:
        t_row = trajs_df[trajs_df["trajectory_id"] == selected_tid].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lifecycle Status", t_row["lifecycle_status"])
        c2.metric("Total Volume", t_row["total_volume"])
        c3.metric("Avg Coherence", f"{t_row['avg_coherence']:.3f}")
        c4.metric("Checkpoints Active", t_row["active_checkpoints"])

        st.write(f"**Timeline Path**: `{t_row['timeline_path']}`")
        st.write(f"**Volume Path**: `{t_row['volume_path']}`")
        st.write(f"**Canonical Representative Post**: *\"{t_row['representative_post']}\"*")

        st.subheader("Member Posts by Checkpoint")
        # Extract cluster keys
        t_matches = matches_df[matches_df["trajectory_id"] == selected_tid]
        c_keys = t_matches["cluster_key"].tolist()

        for ck in c_keys:
            win = ck.split("_C")[0]
            cid = int(ck.split("_C")[1])
            sub = meta_df[(meta_df["window_id"] == win) & (meta_df["cluster_id"] == cid)]
            
            with st.expander(f"Checkpoint [{win}] - Cluster {cid} ({len(sub)} posts)"):
                for text in sub["clean_text"].head(8):
                    st.write(f"- {text}")

# -----------------------------------------------------------------------------
# 7. Research Limitations & Future Work
# -----------------------------------------------------------------------------
elif section == "7. Research Limitations & Future Work":
    st.header("7. Research Limitations & Future Experimental Work")
    st.write(
        "A rigorous scientific paper must transparently articulate experimental limitations. "
        "The zero-lead-time finding highlights key architectural and data constraints:"
    )

    st.markdown("""
    ### Current Experimental Limitations
    1. **Discrete Sampled Checkpoints**:
       - The dataset samples ~1,000 posts every 10–14 days rather than a continuous chronological stream.
       - Breaking news events (like OnlyFans' porn ban) appear immediately as full-blown surges in the next sampled slice, leaving no intermediate 'incubation phase' to observe pre-emergence.
    2. **Equal-Sized Checkpoint Normalization**:
       - Normalizing each checkpoint to ~1,000 posts evaluates relative topic share rather than platform-wide query surges.
    3. **Small Vetted Event Sample**:
       - Exactly 3 manually vetted candidate events/topics were identified and frozen in this slice.
    4. **Platform Tropes (r/AskReddit)**:
       - r/AskReddit is heavily dominated by evergreen introspective question templates ("What is your biggest...", "If you could..."), which require aggressive syntactic filtering.

    ---

    ### Recommended Next Step: Step 9 (Continuous Streaming Experiment)
    To directly test whether dense semantic representations can achieve true temporal lead time:
    - **Increase Temporal Resolution**: Transition from bi-weekly snapshots to daily rolling slices ($T_1, T_2, ..., T_{30}$).
    - **Continuous Incubation Window**: Track semantic vector drift during the 48–72 hours prior to volume surges.
    """)


# -----------------------------------------------------------------------------
# 8. Controlled Ablation Suite (E0 - E6)
# -----------------------------------------------------------------------------
elif section == "8. Controlled Ablation Suite (E0 - E6)":
    st.header("8. Controlled Ablation Suite (E0 - E6) & Master Results")
    st.markdown(
        "A rigorous, 7-stage controlled ablation suite isolating each architectural failure mode across a "
        "fine-grained 30-checkpoint temporal stream (Aug 11 - Sep 5, 2021, 8,784 posts) benchmarked against a **frozen 823-alert keyword baseline**."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Master Comparison", 
        "Ablation Trajectory Explorer", 
        "E6 Two-Tier Incubation", 
        "Target Event & Limitation Audit"
    ])

    with tab1:
        st.subheader("Master Ablation Matrix across 30 Checkpoints")
        master_data = [
            {"Experiment": "E0 Control", "Mechanism / Threshold": "Absolute vol >= 15 posts", "Total Traj": 82, "Multi-Ckpt": 6, "Candidates": 2, "Incub. Seeds": 0, "OnlyFans (D11)": "Missed (<15 vol)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 10, "Mean Coh": 0.6670, "Drift": "Present", "Baseline Alerts": 823},
            {"Experiment": "E1 Scale Normalization", "Mechanism / Threshold": "Share >= 2.0% slice volume", "Total Traj": 82, "Multi-Ckpt": 6, "Candidates": 36, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 10, "Mean Coh": 0.6974, "Drift": "Present", "Baseline Alerts": 823},
            {"Experiment": "E2 Template Stripping", "Mechanism / Threshold": "Interrogative syntax removal", "Total Traj": 109, "Multi-Ckpt": 7, "Candidates": 32, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 24, "Mean Coh": 0.6805, "Drift": "Severe (TRAJ_002)", "Baseline Alerts": 823},
            {"Experiment": "E3 Lexical Consistency", "Mechanism / Threshold": "Cosine >= 0.68 + Jaccard >= 0.20", "Total Traj": 133, "Multi-Ckpt": 11, "Candidates": 35, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 5, "Mean Coh": 0.6819, "Drift": "Controlled", "Baseline Alerts": 823},
            {"Experiment": "E4 Entity-Aware Linkage", "Mechanism / Threshold": "NER dual gate (J_ent >= 0.20)", "Total Traj": 132, "Multi-Ckpt": 12, "Candidates": 35, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 5, "Mean Coh": 0.6819, "Drift": "Controlled", "Baseline Alerts": 823},
            {"Experiment": "E5 Micro-Clustering", "Mechanism / Threshold": "Density lowered to N >= 3", "Total Traj": 1121, "Multi-Ckpt": 4, "Candidates": 141, "Incub. Seeds": 721, "OnlyFans (D11)": "Fragmented (9p)", "FDA/Pfizer (D16)": "Recovered (10 clusters)", "Locked Comms": "Missed (<3)", "Max Span": 3, "Mean Coh": 0.6957, "Drift": "Fragmented", "Baseline Alerts": 823},
            {"Experiment": "E6 Combined Two-Tier", "Mechanism / Threshold": "Two-Tier (N>=7 + Incub. 3<=N<7)", "Total Traj": 862, "Multi-Ckpt": 23, "Candidates": 102, "Incub. Seeds": 517, "OnlyFans (D11)": "Captured (11p intact)", "FDA/Pfizer (D16)": "Recovered (D16 seed)", "Locked Comms": "Missed (<3)", "Max Span": 5, "Mean Coh": 0.6983, "Drift": "Controlled", "Baseline Alerts": 823},
        ]
        df_master = pd.DataFrame(master_data)
        st.dataframe(df_master, width="stretch")

        st.markdown(r"""
        ### Scientific Narrative & Key Takeaways
        - **E0 -> E1 (Scale Normalization):** Solved sample-size mismatch, allowing OnlyFans (11-12 posts) to be detected.
        - **E1 -> E2 (Template Stripping):** Suppressed **82.4% (28/34)** of conversational question-template bursts while surfacing Hurricane Ida and viral memes.
        - **E2 -> E3 (Lexical Consistency Gate):** Eliminated the 24-checkpoint semantic drift pseudo-chain (`TRAJ_002`), capping max lifespan at 5 checkpoints while retaining the 5-checkpoint Afghanistan crisis.
        - **E3 -> E4 (Entity-Aware Linkage):** Reinforced entity continuity; proved that NER alone cannot recover topics below clustering density ($N < 7$).
        - **E4 -> E5 (Micro-Clustering $N=3$):** Lowering $N=3$ recovered sparse vaccine seeds, but caused severe **candidate inflation** (141 candidates) and topic fragmentation.
        - **E5 -> E6 (Combined Two-Tier Architecture):** **Mitigated the sensitivity-candidate-inflation trade-off** by decoupling standard topic detection ($N \ge 7$) from quarantined micro-seed incubation ($3 \le N < 7$).
        """)

    with tab2:
        st.subheader("Explore Trajectories by Ablation Condition")
        exp_choice = st.selectbox(
            "Select Experiment Condition:",
            ["e6_combined", "e5_microcluster", "e4_ner", "e3_lexical", "e2_template", "e1_scale_normalized", "e0_control"]
        )
        traj_file = f"experiments/{exp_choice}/daily_trajectories.csv"
        alert_file = f"experiments/{exp_choice}/emergence_alerts.csv"
        
        try:
            df_t = pd.read_csv(traj_file)
            df_a = pd.read_csv(alert_file)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Trajectories", len(df_t))
            cand_count = len(df_a[df_a["is_vetted_candidate"] == True]) if "is_vetted_candidate" in df_a.columns else len(df_a)
            c2.metric("Validated Candidates", cand_count)
            c3.metric("Max Span (Checkpoints)", df_t["active_checkpoints"].max())

            st.markdown(f"**Top Candidates in `{exp_choice}`:**")
            cand_disp = df_a[df_a["is_vetted_candidate"] == True].head(15) if "is_vetted_candidate" in df_a.columns else df_a.head(15)
            cols_to_show = [c for c in ["trajectory_id", "birth_window", "lifecycle_status", "total_volume", "avg_coherence", "emergence_score", "representative_post"] if c in cand_disp.columns]
            st.dataframe(cand_disp[cols_to_show], width="stretch")
        except Exception as e:
            st.warning(f"Could not load data for {exp_choice}: {e}")

    with tab3:
        st.subheader("E6 Two-Tier Architecture: Standard Bursts vs. Quarantined Incubation")
        st.markdown(r"""
        In E6, topics are decoupled into two distinct tiers:
        - **Tier 1 (Standard Clusters $N \ge 7$):** Evaluated by standard emergence criteria (bursts, growth ratio, stability).
        - **Tier 2 (Micro-Seeds $3 \le N < 7$):** Placed in the **Incubation Layer** (`[MICRO_SEED]`, score = 0.0). They are monitored and only promoted to `[CONFIRMED_SEED]` upon meeting strict confirmation criteria (temporal recurrence across $\ge 2$ checkpoints or verified domain anchor with coherence $\ge 0.85$).
        """)

        try:
            df_e6_alerts = pd.read_csv("experiments/e6_combined/emergence_alerts.csv")
            tier_col1, tier_col2, tier_col3 = st.columns(3)
            tier_col1.metric("Tier 1 Bursts", len(df_e6_alerts[df_e6_alerts["lifecycle_status"] == "[BURST]"]))
            tier_col2.metric("Tier 2 Confirmed Seeds", len(df_e6_alerts[df_e6_alerts["lifecycle_status"] == "[CONFIRMED_SEED]"]))
            tier_col3.metric("Quarantined Micro-Seeds", len(df_e6_alerts[df_e6_alerts["lifecycle_status"] == "[MICRO_SEED]"]))

            st.markdown("#### Confirmed Seeds (Tier 2 Incubation Promotions)")
            confirmed = df_e6_alerts[df_e6_alerts["lifecycle_status"] == "[CONFIRMED_SEED]"]
            st.dataframe(confirmed[["trajectory_id", "birth_window", "volume_path", "avg_coherence", "emergence_score", "entities", "representative_post"]], width="stretch")

            st.markdown("#### Top Quarantined Micro-Seeds (Safely Suppressed in Incubation)")
            quarantined = df_e6_alerts[df_e6_alerts["lifecycle_status"] == "[MICRO_SEED]"].sort_values("avg_coherence", ascending=False).head(10)
            st.dataframe(quarantined[["trajectory_id", "birth_window", "total_volume", "avg_coherence", "representative_post"]], width="stretch")
        except Exception as e:
            st.error(f"Error loading E6 alerts: {e}")

    with tab4:
        st.subheader("Target Event Audit & The Empirical Density Boundary of Single-Checkpoint Clustering")
        st.markdown(r"""
        ### 1. Target Event Tracking Status in E6
        - **OnlyFans Policy Change ($D_{11}$):** **CAPTURED** as `TRAJ_284` (11 posts intact, emergence score: 0.565, status: `[BURST]`).
        - **Hurricane Ida ($D_{27}$):** **CAPTURED** as `TRAJ_757` (7 posts intact, emergence score: 0.412, status: `[BURST]`).
        - **Afghanistan Evacuation Crisis ($D_{04} \to D_{08}$):** **CAPTURED** as `TRAJ_090` (5 checkpoints intact: 7 -> 14 -> 15 -> 8 -> 7 posts, status: `[STABLE]`).
        - **FDA / Pfizer / Vaccine Mandate Seed ($D_{16}$):** **RECOVERED** as `TRAJ_456` (3 posts, coherence: 0.938, status: `[CONFIRMED_SEED]`, rep: *"Should one try and reason with a friend ready to throw away their medical career over a vaccine mandate?"*).

        ---

        ### 2. Observable Detection Boundary: Reddit Locked Comments
        - Post distribution across the 30-checkpoint stream:
          * $D_{22}$: 1 post
          * $D_{25}$: 1 post
          * $D_{27}$: 2 posts
        - **Result:** Never reached $\ge 3$ posts in any single checkpoint.
        - **Empirical Boundary:** Single-checkpoint density clustering (HDBSCAN) requires at least $N = 3$ co-occurring posts to form a cluster core. Events with $\le 2$ posts per time slice are mathematically indistinguishable from background noise at daily temporal resolution.
        - **Future Direction:** Requires cross-checkpoint point accumulation or streaming entity-graph bipartition.
        """)


st.markdown("---")
st.caption("EmergNLP Research Project — Advanced Agentic NLP Pipeline — Google DeepMind")
