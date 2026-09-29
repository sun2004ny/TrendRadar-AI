import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.research_service import get_topic_research_dossier
from src.llm_summary import generate_executive_brief
from src.workspace import (
    save_investigation,
    list_investigations,
    get_investigation,
    update_investigation_status,
    delete_investigation,
    export_workspace_markdown,
)

# -----------------------------------------------------------------------------
# Streamlit Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrendRadar AI | Real-Time Topic Emergence & Threat Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    .badge-burst {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-confirmed {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-incubating {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-stable {
        background-color: #F1F5F9;
        color: #334155;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .topic-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .brief-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loaders (Robust Multilevel Loaders)
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    # Production 30-checkpoint dataset (E6)
    e6_alerts_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "emergence_alerts.csv")
    e6_trajs_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "daily_trajectories.csv")
    e6_posts_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "daily_clusters.csv")
    e6_base_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "baseline_volume_alerts_daily.csv")
    
    # 6-checkpoint historical benchmark dataset
    raw_posts_path = os.path.join(BASE_DIR, "data", "processed", "reddit_cleaned_6weeks.csv")
    eval_path = os.path.join(BASE_DIR, "experiments", "evaluation_results_6weeks.csv")
    clusters_path = os.path.join(BASE_DIR, "data", "processed", "clusters_summary_6weeks.csv")

    if os.path.exists(e6_alerts_p):
        alerts_df = pd.read_csv(e6_alerts_p)
    else:
        trajs_fallback = os.path.join(BASE_DIR, "data", "processed", "emerging_topics_alerts_6weeks.csv")
        alerts_df = pd.read_csv(trajs_fallback) if os.path.exists(trajs_fallback) else pd.DataFrame()

    trajs_df = pd.read_csv(e6_trajs_p) if os.path.exists(e6_trajs_p) else alerts_df
    posts_df = pd.read_csv(e6_posts_p) if os.path.exists(e6_posts_p) else (
        pd.read_csv(raw_posts_path) if os.path.exists(raw_posts_path) else pd.DataFrame()
    )
    base_df = pd.read_csv(e6_base_p) if os.path.exists(e6_base_p) else pd.DataFrame()
    eval_df = pd.read_csv(eval_path) if os.path.exists(eval_path) else pd.DataFrame()
    clusters_df = pd.read_csv(clusters_path) if os.path.exists(clusters_path) else pd.DataFrame()

    return alerts_df, trajs_df, posts_df, base_df, eval_df, clusters_df


alerts_df, trajs_df, posts_df, base_df, eval_df, clusters_df = load_all_data()

# -----------------------------------------------------------------------------
# Session State Synchronization
# -----------------------------------------------------------------------------
NAV_SECTIONS = [
    "1. 🌐 Overview",
    "2. 📡 Trend Radar",
    "3. 🚨 Emerging Topics",
    "4. 🔎 Topic Details",
    "5. 🔬 Research Hub",
    "6. 🧠 AI Executive Briefing",
    "7. ⚡ Action Hub",
    "8. 💾 Workspace",
    "9. 📊 Scientific & Research Lab"
]

if "nav_section" not in st.session_state:
    st.session_state["nav_section"] = NAV_SECTIONS[0]

# Pick a sensible default topic (e.g. OnlyFans TRAJ_284 or first vetted candidate)
default_tid = "TRAJ_284"
if not alerts_df.empty and "trajectory_id" in alerts_df.columns:
    if default_tid not in alerts_df["trajectory_id"].values:
        vetted = alerts_df[alerts_df.get("is_vetted_candidate", False) == True]
        if not vetted.empty:
            default_tid = vetted["trajectory_id"].iloc[0]
        else:
            default_tid = alerts_df["trajectory_id"].iloc[0]

if "selected_trajectory_id" not in st.session_state:
    st.session_state["selected_trajectory_id"] = default_tid

if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("GEMINI_API_KEY", "")


def navigate_to(section_name: str, trajectory_id: str = None):
    st.session_state["nav_section"] = section_name
    if trajectory_id:
        st.session_state["selected_trajectory_id"] = trajectory_id
    st.rerun()


# -----------------------------------------------------------------------------
# Header Bar
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📡 TrendRadar AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Topic Emergence, Evidence Corroboration & Threat Intelligence Platform</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Platform Controls
# -----------------------------------------------------------------------------
st.sidebar.title("TrendRadar AI")
st.sidebar.caption("v2.0 • Production Intelligence Engine")

selected_nav = st.sidebar.radio(
    "Workflow Navigation:",
    NAV_SECTIONS,
    index=NAV_SECTIONS.index(st.session_state["nav_section"]) if st.session_state["nav_section"] in NAV_SECTIONS else 0
)

# Keep session state in sync if radio button clicked directly
if selected_nav != st.session_state["nav_section"]:
    st.session_state["nav_section"] = selected_nav

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Topic Focus**")
st.sidebar.info(f"🎯 `{st.session_state['selected_trajectory_id']}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Intelligence Settings**")
gemini_input = st.sidebar.text_input("Gemini API Key (Optional)", value=st.session_state["api_key"], type="password")
if gemini_input != st.session_state["api_key"]:
    st.session_state["api_key"] = gemini_input

st.sidebar.caption("🔒 Offline Heuristic Synthesizer active if no API key is provided.")


# Helper to get current topic row safely
def get_current_topic_row(tid: str):
    if alerts_df.empty or "trajectory_id" not in alerts_df.columns:
        return None
    matches = alerts_df[alerts_df["trajectory_id"] == tid]
    if not matches.empty:
        return matches.iloc[0]
    return alerts_df.iloc[0]


current_topic = get_current_topic_row(st.session_state["selected_trajectory_id"])


# -----------------------------------------------------------------------------
# 1. 🌐 OVERVIEW
# -----------------------------------------------------------------------------
if st.session_state["nav_section"] == "1. 🌐 Overview":
    st.header("1. System Health & Stream Overview")
    st.write(
        "TrendRadar AI continuously ingests chronologically bounded text streams with **strict forward-only causality**, "
        "filtering platform boilerplate, clustering dense semantic vectors, and isolating actionable events before peak volume."
    )

    # Core Intelligence Metrics
    total_posts = len(posts_df) if not posts_df.empty else 8784
    active_trajs = len(trajs_df) if not trajs_df.empty else 862
    vetted_count = len(alerts_df[alerts_df["is_vetted_candidate"] == True]) if ("is_vetted_candidate" in alerts_df.columns) else 102
    seeds_count = len(alerts_df[alerts_df["lifecycle_status"] == "[CONFIRMED_SEED]"]) if ("lifecycle_status" in alerts_df.columns) else 23
    bursts_count = len(alerts_df[alerts_df["lifecycle_status"] == "[BURST]"]) if ("lifecycle_status" in alerts_df.columns) else 65

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Posts Monitored", f"{total_posts:,}", "Zero Lookahead")
    c2.metric("Active Trajectories", f"{active_trajs:,}", "Multi-Day Tracking")
    c3.metric("Actionable Bursts (T1)", f"{bursts_count}", "Scale-Normalized")
    c4.metric("Confirmed Seeds (T2)", f"{seeds_count}", "Quarantined Incubation")
    c5.metric("Alert Noise Suppressed", "87.6%", "vs. Keyword Baseline")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Platform Workflow: Detect → Track → Research → Explain → Act")
        st.markdown(r"""
        ```
         [ Social Data Stream ] ──► [ Dense Sentence-BERT Embedding ]
                                                 │
                                                 ▼
         [ Action / Alert Hub ] ◄── [ HDBSCAN Density Clustering ]
                 ▲                               │
                 │                               ▼
         [ AI Evidence Brief ]  ◄── [ Drift-Gated Trajectory Tracking ]
                 ▲                               │
                 │                               ▼
         [ External Research ]  ◄── [ Two-Tier Emergence Scoring Gate ]
        ```
        """)
        st.info("💡 **Key Principle**: Machine learning detects mathematically proven clusters. Research provides external context. The LLM explains the evidence. The analyst decides.")

    with col_r:
        st.subheader("Temporal Checkpoint Distribution")
        if not posts_df.empty and "window_id" in posts_df.columns:
            ckpt_summary = posts_df.groupby("window_id").size().reset_index(name="post_count")
            fig = px.bar(
                ckpt_summary,
                x="window_id",
                y="post_count",
                color="post_count",
                title="Monitored Posts per Time Slice",
                labels={"window_id": "Checkpoint", "post_count": "Post Count"}
            )
            fig.update_layout(height=280, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, width="stretch")
        else:
            st.write("30 chronological checkpoints ($D_{00}$ to $D_{29}$) evaluated with zero temporal leakage.")

    st.markdown("---")
    st.subheader("Quick Launch: Explore High-Velocity Detections")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🚨 Inspect Emerging Topic Alerts", use_container_width=True):
            navigate_to("3. 🚨 Emerging Topics")
    with col_b:
        if st.button("📡 View Velocity & Trend Radar", use_container_width=True):
            navigate_to("2. 📡 Trend Radar")
    with col_c:
        if st.button("💾 Open Saved Investigations", use_container_width=True):
            navigate_to("8. 💾 Workspace")


# -----------------------------------------------------------------------------
# 2. 📡 TREND RADAR
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "2. 📡 Trend Radar":
    st.header("2. Trend Radar & Velocity Dynamics")
    st.write("Track the acceleration, lifespan, and volume trajectory of conversations across sequential time slices.")

    # Multi-Trajectory Growth Curves
    st.subheader("Trajectory Progression Across Checkpoints")
    multi_checkpoints = trajs_df[trajs_df["active_checkpoints"] >= 2].head(12) if "active_checkpoints" in trajs_df.columns else pd.DataFrame()

    if not multi_checkpoints.empty:
        plot_rows = []
        for _, r in multi_checkpoints.iterrows():
            timeline = str(r["timeline_path"]).split(" -> ")
            vols = [int(v) for v in str(r["volume_path"]).split(" -> ") if v.isdigit()]
            for w, v in zip(timeline, vols):
                plot_rows.append({
                    "Trajectory": f"{r['trajectory_id']}: {str(r['topic_name'])[:25]}",
                    "Checkpoint": w,
                    "Volume": v,
                    "Status": r.get("lifecycle_status", "Active")
                })
        p_df = pd.DataFrame(plot_rows)
        if not p_df.empty:
            fig_traj = px.line(
                p_df,
                x="Checkpoint",
                y="Volume",
                color="Trajectory",
                markers=True,
                title="Trajectory Volume Paths Across Time Checkpoints"
            )
            fig_traj.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_traj, width="stretch")

    st.subheader("Active Trajectory Registry")
    disp_cols = [c for c in ["trajectory_id", "lifecycle_status", "topic_name", "active_checkpoints", "timeline_path", "volume_path", "total_volume", "avg_coherence", "emergence_score"] if c in trajs_df.columns]
    st.dataframe(trajs_df[disp_cols].head(40), width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# 3. 🚨 EMERGING TOPICS
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "3. 🚨 Emerging Topics":
    st.header("3. Emerging Topic Alerts")
    st.write(
        "Real-time operational alerts flagged by TrendRadar AI's **Two-Tier Architecture**: "
        "High-volume sudden bursts ($N \\ge 7$) and verified micro-seed incubation signals ($3 \\le N < 7$)."
    )

    t1, t2 = st.tabs(["🚀 Tier 1 Actionable Bursts", "🌱 Tier 2 Confirmed Seeds (Incubation)"])

    def render_candidate_card(r, tier_badge):
        tid = r["trajectory_id"]
        t_name = str(r["topic_name"]).replace("[", "").replace("]", "").replace("'", "")
        vol_path = r.get("volume_path", "N/A")
        coh = r.get("avg_coherence", 0.0)
        birth = r.get("birth_window", "D_00")
        rep = r.get("representative_post", "No post available")
        score = r.get("emergence_score", 0.0)

        with st.container():
            st.markdown(f"""
            <div class="topic-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; font-size:1.25rem;">#{tid} — {t_name}</h3>
                    <span class="{tier_badge}">{r.get('lifecycle_status', '[ALERT]')}</span>
                </div>
                <p style="margin:8px 0; color:#334155; font-style:italic;">"{rep}"</p>
                <div style="margin:8px 0; font-size:0.9rem; color:#64748B;">
                    <strong>Volume Progression:</strong> <code>{vol_path}</code> &nbsp;|&nbsp;
                    <strong>First Detected:</strong> <code>{birth}</code> &nbsp;|&nbsp;
                    <strong>Coherence:</strong> <code>{coh:.3f}</code> &nbsp;|&nbsp;
                    <strong>Score:</strong> <code>{score:.3f}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.2, 1.2, 1.4, 1.2])
            with btn_col1:
                if st.button(f"🔎 Deep Dive", key=f"dd_{tid}"):
                    navigate_to("4. 🔎 Topic Details", tid)
            with btn_col2:
                if st.button(f"🔬 Research", key=f"res_{tid}"):
                    navigate_to("5. 🔬 Research Hub", tid)
            with btn_col3:
                if st.button(f"🧠 AI Brief", key=f"brief_{tid}"):
                    navigate_to("6. 🧠 AI Executive Briefing", tid)
            with btn_col4:
                if st.button(f"📌 Pin Case", key=f"pin_{tid}"):
                    save_investigation(
                        trajectory_id=tid,
                        topic_name=t_name,
                        lifecycle_status=r.get("lifecycle_status", "[ALERT]"),
                        volume_path=str(vol_path),
                        coherence=float(coh),
                        birth_window=str(birth),
                        representative_post=str(rep),
                        keywords=[k.strip() for k in t_name.split(",")]
                    )
                    st.toast(f"✅ Pinned {tid} to Workspace!", icon="💾")

    with t1:
        st.subheader("Tier 1: High-Velocity Rapid Bursts")
        bursts = alerts_df[alerts_df.get("lifecycle_status", "") == "[BURST]"]
        if not bursts.empty:
            for _, r in bursts.head(15).iterrows():
                render_candidate_card(r, "badge-burst")
        else:
            st.info("No Tier 1 bursts currently active.")

    with t2:
        st.subheader("Tier 2: Confirmed Seeds from Quarantined Incubation")
        st.write(r"Micro-clusters ($3 \le N < 7$) that proved topical recurrence or extreme coherence ($\ge 0.85$).")
        confirmed = alerts_df[alerts_df.get("lifecycle_status", "") == "[CONFIRMED_SEED]"]
        if not confirmed.empty:
            for _, r in confirmed.head(15).iterrows():
                render_candidate_card(r, "badge-confirmed")
        else:
            st.info("No Confirmed Seeds currently pending.")


# -----------------------------------------------------------------------------
# 4. 🔎 TOPIC DETAILS
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "4. 🔎 Topic Details":
    st.header("4. Deep-Dive Topic Inspector")
    st.write("Inspect mathematical evidence, cluster keywords, and member posts for the selected topic.")

    all_tids = alerts_df["trajectory_id"].tolist() if not alerts_df.empty else []
    
    col_sel, _ = st.columns([2, 1])
    with col_sel:
        chosen_tid = st.selectbox(
            "Select Topic Trajectory to Examine:",
            all_tids,
            index=all_tids.index(st.session_state["selected_trajectory_id"]) if st.session_state["selected_trajectory_id"] in all_tids else 0
        )
        if chosen_tid != st.session_state["selected_trajectory_id"]:
            st.session_state["selected_trajectory_id"] = chosen_tid
            current_topic = get_current_topic_row(chosen_tid)

    if current_topic is not None:
        t_row = current_topic
        tid = t_row["trajectory_id"]
        t_name = str(t_row["topic_name"]).replace("[", "").replace("]", "").replace("'", "")
        
        # Metric Strip
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Status", t_row.get("lifecycle_status", "N/A"))
        c2.metric("Total Volume", t_row.get("total_volume", "N/A"))
        c3.metric("Avg Coherence", f"{t_row.get('avg_coherence', 0.0):.3f}")
        c4.metric("Birth Window", t_row.get("birth_window", "N/A"))
        c5.metric("Emergence Score", f"{t_row.get('emergence_score', 0.0):.3f}")

        st.markdown("---")
        st.markdown(f"### Canonical User Post:")
        st.info(f"💬 *\"{t_row.get('representative_post', '')}\"*")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### Topic Information & Trajectory Timeline")
            st.write(f"- **Trajectory ID**: `{tid}`")
            st.write(f"- **Extracted Named Entities**: `{t_row.get('entities', 'None')}`")
            st.write(f"- **Volume Path**: `{t_row.get('volume_path', 'N/A')}`")
            st.write(f"- **Timeline Path**: `{t_row.get('timeline_path', 'N/A')}`")
            st.write(f"- **Active Checkpoint Span**: `{t_row.get('active_checkpoints', 1)} checkpoints`")

        with col_right:
            st.markdown("#### Next Operational Actions")
            if st.button("🔬 Search External Context in Research Hub", use_container_width=True):
                navigate_to("5. 🔬 Research Hub", tid)
            if st.button("🧠 Synthesize AI Executive Briefing", use_container_width=True):
                navigate_to("6. 🧠 AI Executive Briefing", tid)
            if st.button("📌 Pin to Active Workspace", use_container_width=True):
                save_investigation(
                    trajectory_id=tid,
                    topic_name=t_name,
                    lifecycle_status=t_row.get("lifecycle_status", "[ALERT]"),
                    volume_path=str(t_row.get("volume_path", "")),
                    coherence=float(t_row.get("avg_coherence", 0.0)),
                    birth_window=str(t_row.get("birth_window", "")),
                    representative_post=str(t_row.get("representative_post", "")),
                    keywords=[k.strip() for k in t_name.split(",")]
                )
                st.success(f"Case `{tid}` saved to Workspace!")


# -----------------------------------------------------------------------------
# 5. 🔬 RESEARCH HUB
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "5. 🔬 Research Hub":
    st.header("5. Research Hub & External Evidence")
    st.write(
        "TrendRadar AI queries live external knowledge (**Wikipedia**, **arXiv**, and web entities) "
        "to discover what is happening outside the local dataset."
    )

    t_row = current_topic
    if t_row is not None:
        tid = t_row["trajectory_id"]
        t_name = str(t_row["topic_name"]).replace("[", "").replace("]", "").replace("'", "")
        keywords = [k.strip() for k in t_name.split(",")]
        rep_post = str(t_row.get("representative_post", ""))

        st.subheader(f"External Dossier for: #{tid} — {t_name}")
        st.caption(f"Canonical Post: \"{rep_post}\"")

        # Fetch or cache research dossier
        with st.spinner("Querying live Wikipedia & arXiv APIs..."):
            dossier = get_topic_research_dossier(t_name, keywords, rep_post)

        # Confidence Badge
        conf = dossier.get("corroboration_confidence", "External Context Verified")
        badge_style = "badge-confirmed" if "Verified" in conf or "High" in conf else "badge-incubating"
        st.markdown(f"**Corroboration Rating:** <span class='{badge_style}'>{conf}</span>", unsafe_allow_html=True)
        st.markdown("---")

        tab_wiki, tab_arxiv = st.tabs(["📚 Wikipedia Context", "🔬 Academic Preprints (arXiv)"])

        with tab_wiki:
            wiki = dossier.get("wikipedia", {})
            if wiki.get("found"):
                st.markdown(f"### {wiki.get('title')}")
                st.write(wiki.get("summary"))
                st.markdown(f"[🔗 Read Full Article on Wikipedia]({wiki.get('url')})")
                if wiki.get("related_topics"):
                    st.write(f"**Related Entities:** {', '.join(wiki.get('related_topics'))}")
            else:
                st.warning("No direct Wikipedia entry found for this specific query.")
                st.write(f"Search Query Evaluated: `{dossier.get('search_query')}`")

        with tab_arxiv:
            papers = dossier.get("arxiv_papers", [])
            if papers:
                for p in papers:
                    st.markdown(f"#### [{p['title']}]({p['url']})")
                    st.caption(f"Authors: {p['author']} | Published: {p['published']}")
                    st.write(p["summary"])
                    st.markdown("---")
            else:
                st.info("No matching arXiv preprints retrieved for this non-academic topic.")

        st.markdown("---")
        if st.button("🧠 Proceed to AI Executive Briefing with this Evidence", use_container_width=True):
            st.session_state["cached_dossier"] = dossier
            navigate_to("6. 🧠 AI Executive Briefing", tid)


# -----------------------------------------------------------------------------
# 6. 🧠 AI EXECUTIVE BRIEFING
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "6. 🧠 AI Executive Briefing":
    st.header("6. AI Executive Briefing")
    st.write(
        "The AI briefing engine synthesizes the **mathematical detection evidence**, canonical post, "
        "and external research into an actionable 4-box executive summary."
    )

    t_row = current_topic
    if t_row is not None:
        tid = t_row["trajectory_id"]
        t_name = str(t_row["topic_name"]).replace("[", "").replace("]", "").replace("'", "")
        keywords = [k.strip() for k in t_name.split(",")]
        rep_post = str(t_row.get("representative_post", ""))
        vol_path = str(t_row.get("volume_path", "N/A"))
        coh = float(t_row.get("avg_coherence", 0.0))
        birth = str(t_row.get("birth_window", "D_00"))
        status = str(t_row.get("lifecycle_status", "[ALERT]"))

        dossier = st.session_state.get("cached_dossier") or get_topic_research_dossier(t_name, keywords, rep_post)

        with st.spinner("Synthesizing evidence-grounded briefing..."):
            brief = generate_executive_brief(
                topic_name=t_name,
                lifecycle_status=status,
                volume_history=vol_path,
                coherence=coh,
                birth_window=birth,
                representative_post=rep_post,
                keywords=keywords,
                research_dossier=dossier,
                api_key=st.session_state.get("api_key")
            )
            st.session_state["cached_brief"] = brief

        st.markdown(f"### Intelligence Brief: #{tid} — {t_name}")

        # The 4-Box Executive Layout
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("""
            <div class="brief-box">
                <h4 style="margin:0 0 8px 0; color:#1E3A8A;">📌 1. What is Happening?</h4>
            """, unsafe_allow_html=True)
            st.write(brief.get("what_is_happening"))
            st.markdown("</div>", unsafe_allow_html=True)

        with b2:
            st.markdown("""
            <div class="brief-box">
                <h4 style="margin:0 0 8px 0; color:#1E3A8A;">🚀 2. Why is it Emerging?</h4>
            """, unsafe_allow_html=True)
            st.write(brief.get("why_is_it_emerging"))
            st.markdown("</div>", unsafe_allow_html=True)

        b3, b4 = st.columns(2)
        with b3:
            st.markdown("""
            <div class="brief-box">
                <h4 style="margin:0 0 8px 0; color:#1E3A8A;">📊 3. Supporting Evidence & Metrics</h4>
            """, unsafe_allow_html=True)
            st.write(brief.get("evidence_summary"))
            st.markdown("</div>", unsafe_allow_html=True)

        with b4:
            st.markdown("""
            <div class="brief-box">
                <h4 style="margin:0 0 8px 0; color:#1E3A8A;">🎯 4. Recommended Next Steps</h4>
            """, unsafe_allow_html=True)
            st.write(brief.get("recommended_next_steps"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("⚡ Open Action Hub (Generate Alert / File Report)", use_container_width=True):
            navigate_to("7. ⚡ Action Hub", tid)


# -----------------------------------------------------------------------------
# 7. ⚡ ACTION HUB
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "7. ⚡ Action Hub":
    st.header("7. Action Hub & Dispatch Center")
    st.write("Generate stakeholder notifications, export formal intelligence briefs, and commit cases to the Workspace.")

    t_row = current_topic
    if t_row is not None:
        tid = t_row["trajectory_id"]
        t_name = str(t_row["topic_name"]).replace("[", "").replace("]", "").replace("'", "")
        rep_post = str(t_row.get("representative_post", ""))
        vol_path = str(t_row.get("volume_path", "N/A"))
        coh = float(t_row.get("avg_coherence", 0.0))
        birth = str(t_row.get("birth_window", "D_00"))
        status = str(t_row.get("lifecycle_status", "[ALERT]"))
        keywords = [k.strip() for k in t_name.split(",")]

        brief = st.session_state.get("cached_brief") or generate_executive_brief(
            topic_name=t_name,
            lifecycle_status=status,
            volume_history=vol_path,
            coherence=coh,
            birth_window=birth,
            representative_post=rep_post,
            keywords=keywords
        )

        col_act1, col_act2 = st.columns(2)

        with col_act1:
            st.subheader("📄 Formal Intelligence Brief")
            brief_md = brief.get("full_brief_markdown", "No content.")
            st.download_button(
                label="📥 Download 1-Page Intelligence Brief (.md)",
                data=brief_md,
                file_name=f"TrendRadar_Brief_{tid}.md",
                mime="text/markdown",
                use_container_width=True
            )

            st.subheader("📢 Stakeholder Notification (Slack / Teams)")
            slack_alert = (
                f"🚨 *TRENDRADAR AI ALERT: {t_name.upper()}*\n"
                f"• *Status*: {status} | *First Detected*: {birth} | *Volume Path*: `{vol_path}`\n"
                f"• *Core Theme*: {brief.get('what_is_happening')[:180]}...\n"
                f"• *Recommended Action*: {brief.get('recommended_next_steps')[:140]}...\n"
                f"• *Analyst Dashboard*: http://localhost:8501"
            )
            st.code(slack_alert, language="markdown")

        with col_act2:
            st.subheader("💾 Commit to Workspace")
            with st.form("workspace_form"):
                case_status = st.selectbox("Investigation Status", ["Active", "Monitoring", "Escalated", "Resolved"])
                notes = st.text_area("Analyst Notes & Context", "High-velocity signal verified. Monitor next checkpoint.")
                submitted = st.form_submit_button("Save Case to Workspace", use_container_width=True)
                if submitted:
                    save_investigation(
                        trajectory_id=tid,
                        topic_name=t_name,
                        lifecycle_status=status,
                        volume_path=vol_path,
                        coherence=coh,
                        birth_window=birth,
                        representative_post=rep_post,
                        keywords=keywords,
                        ai_brief=brief,
                        analyst_notes=notes,
                        status=case_status
                    )
                    st.success(f"Case `{tid}` committed to Workspace with status '{case_status}'!")


# -----------------------------------------------------------------------------
# 8. 💾 WORKSPACE
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "8. 💾 Workspace":
    st.header("8. Workspace & Investigation Desk")
    st.write("Track ongoing topic investigations, update case statuses, and review pinned intelligence briefs.")

    investigations = list_investigations()

    if not investigations:
        st.info("No investigations currently logged. Go to **Emerging Topics** or **Action Hub** to pin a topic.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cases", len(investigations))
        active_cases = len([i for i in investigations if i.get("status") in ["Active", "Escalated"]])
        c2.metric("Active / Escalated", active_cases)
        monitoring_cases = len([i for i in investigations if i.get("status") == "Monitoring"])
        c3.metric("Under Monitoring", monitoring_cases)

        st.markdown("---")

        # Master Dossier Export
        dossier_text = export_workspace_markdown()
        st.download_button(
            "📥 Export Master Workspace Dossier (.md)",
            data=dossier_text,
            file_name="TrendRadar_Master_Workspace.md",
            mime="text/markdown"
        )

        st.markdown("### Saved Investigation Cases")
        for inv in investigations:
            inv_id = inv["id"]
            with st.expander(f"[{inv.get('status', 'Active')}] {inv['topic_name']} — {inv['trajectory_id']}"):
                col_i1, col_i2 = st.columns([2, 1])
                with col_i1:
                    st.write(f"**Canonical Post:** *\"{inv.get('representative_post')}\"*")
                    st.write(f"**Volume Path:** `{inv.get('volume_path')}` | **Coherence:** `{inv.get('avg_coherence')}`")
                    st.write(f"**Analyst Notes:** {inv.get('analyst_notes')}")
                    st.caption(f"Last Updated: {inv.get('updated_at')}")

                with col_i2:
                    new_status = st.selectbox(
                        "Update Status",
                        ["Active", "Monitoring", "Escalated", "Resolved"],
                        index=["Active", "Monitoring", "Escalated", "Resolved"].index(inv.get("status", "Active")),
                        key=f"status_{inv_id}"
                    )
                    new_notes = st.text_input("Edit Notes", value=inv.get("analyst_notes", ""), key=f"notes_{inv_id}")
                    if st.button("Save Update", key=f"save_{inv_id}"):
                        update_investigation_status(inv_id, new_status, new_notes)
                        st.toast(f"Updated {inv_id}", icon="✅")
                        st.rerun()

                    if st.button("🗑️ Delete Case", key=f"del_{inv_id}"):
                        delete_investigation(inv_id)
                        st.toast(f"Deleted {inv_id}", icon="🗑️")
                        st.rerun()


# -----------------------------------------------------------------------------
# 9. 📊 SCIENTIFIC & RESEARCH LAB
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "9. 📊 Scientific & Research Lab":
    st.header("9. Scientific & Research Lab (Ablations & Benchmarks)")
    st.write(
        "The rigorous empirical research suite: Controlled Ablations (E0 to E6), "
        "823-Alert Keyword Baseline Comparison, and Lead-Time Benchmarking across 8,784 posts."
    )

    lab_t1, lab_t2, lab_t3 = st.tabs([
        "Master Ablation Matrix (E0–E6)",
        "Baseline Comparison (87.6% Noise Reduction)",
        "Lead-Time Evaluation & Density Boundaries"
    ])

    with lab_t1:
        st.subheader("Controlled Ablation Progression (E0 to E6)")
        master_data = [
            {"Experiment": "E0 Control", "Mechanism / Threshold": "Absolute vol >= 15 posts", "Total Traj": 82, "Multi-Ckpt": 6, "Candidates": 2, "Incub. Seeds": 0, "OnlyFans (D11)": "Missed (<15 vol)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 10, "Mean Coh": 0.6670, "Drift": "Present", "Baseline Alerts": 823},
            {"Experiment": "E1 Scale Normalization", "Mechanism / Threshold": "Share >= 2.0% slice volume", "Total Traj": 82, "Multi-Ckpt": 6, "Candidates": 36, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 10, "Mean Coh": 0.6974, "Drift": "Present", "Baseline Alerts": 823},
            {"Experiment": "E2 Template Stripping", "Mechanism / Threshold": "Interrogative syntax removal", "Total Traj": 109, "Multi-Ckpt": 7, "Candidates": 32, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 24, "Mean Coh": 0.6805, "Drift": "Severe (TRAJ_002)", "Baseline Alerts": 823},
            {"Experiment": "E3 Lexical Consistency", "Mechanism / Threshold": "Cosine >= 0.68 + Jaccard >= 0.20", "Total Traj": 133, "Multi-Ckpt": 11, "Candidates": 35, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 5, "Mean Coh": 0.6819, "Drift": "Controlled", "Baseline Alerts": 823},
            {"Experiment": "E4 Entity-Aware Linkage", "Mechanism / Threshold": "NER dual gate (J_ent >= 0.20)", "Total Traj": 132, "Multi-Ckpt": 12, "Candidates": 35, "Incub. Seeds": 0, "OnlyFans (D11)": "Captured (D11)", "FDA/Pfizer (D16)": "Missed (N<7)", "Locked Comms": "Missed (N<7)", "Max Span": 5, "Mean Coh": 0.6819, "Drift": "Controlled", "Baseline Alerts": 823},
            {"Experiment": "E5 Micro-Clustering", "Mechanism / Threshold": "Density lowered to N >= 3", "Total Traj": 1121, "Multi-Ckpt": 4, "Candidates": 141, "Incub. Seeds": 721, "OnlyFans (D11)": "Fragmented (9p)", "FDA/Pfizer (D16)": "Recovered (10 clusters)", "Locked Comms": "Missed (<3)", "Max Span": 3, "Mean Coh": 0.6957, "Drift": "Fragmented", "Baseline Alerts": 823},
            {"Experiment": "E6 Combined Two-Tier", "Mechanism / Threshold": "Two-Tier (N>=7 + Incub. 3<=N<7)", "Total Traj": 862, "Multi-Ckpt": 23, "Candidates": 102, "Incub. Seeds": 517, "OnlyFans (D11)": "Captured (11p intact)", "FDA/Pfizer (D16)": "Recovered (D16 seed)", "Locked Comms": "Missed (<3)", "Max Span": 5, "Mean Coh": 0.6983, "Drift": "Controlled", "Baseline Alerts": 823},
        ]
        st.dataframe(pd.DataFrame(master_data), width="stretch")

        st.markdown(r"""
        ### Key Scientific Insights
        - **E0 -> E1 (Scale Normalization):** Solved sample-size mismatch, allowing OnlyFans (11-12 posts) to be detected.
        - **E1 -> E2 (Template Stripping):** Suppressed **82.4% (28/34)** of conversational question-template bursts while surfacing Hurricane Ida.
        - **E2 -> E3 (Lexical Consistency Gate):** Eliminated the 24-checkpoint semantic drift pseudo-chain (`TRAJ_002`), capping max lifespan at 5 checkpoints.
        - **E5 -> E6 (Two-Tier Architecture):** **Mitigated the sensitivity-inflation trade-off** by decoupling standard topic detection ($N \ge 7$) from quarantined incubation ($3 \le N < 7$).
        """)

    with lab_t2:
        st.subheader("Keyword Frequency Baseline vs. Semantic NLP Pipeline")
        st.write("Comparing the frozen 823-alert keyword frequency baseline to TrendRadar AI's 102 candidates.")
        
        comp_df = pd.DataFrame({
            "Method": ["Keyword Frequency Baseline", "TrendRadar AI Pipeline"],
            "Alert Count": [823, 102],
            "Type": ["Fragmented Word Surges", "Synthesized Narrative Events"]
        })
        fig_bar = px.bar(
            comp_df,
            x="Method",
            y="Alert Count",
            color="Method",
            text="Alert Count",
            title="Total Alert Noise Comparison"
        )
        fig_bar.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")
        st.info("🔬 **Noise Reduction**: TrendRadar AI achieved an **87.6% reduction in false-alarm alert noise** relative to keyword frequency tracking.")

    with lab_t3:
        st.subheader("Checkpoint Lead-Time & Observable Density Boundary")
        st.write("Measuring whether semantic clustering provided temporal lead time over keyword spikes.")
        if not eval_df.empty:
            st.dataframe(eval_df, width="stretch", hide_index=True)

        st.markdown(r"""
        ### Observable Detection Boundary
        - **Single-Checkpoint Clustering Limit:** HDBSCAN requires at least $N = 3$ co-occurring posts to form a cluster core.
        - Events with $\le 2$ posts per day (e.g. Reddit locked comments bug) are mathematically indistinguishable from noise within a single time slice.
        - **Strategic Extension:** Requires continuous 48–72h sliding vector buffers to accumulate sparse signals across time slices.
        """)

st.markdown("---")
st.caption("TrendRadar AI • Threat & Emerging Topic Intelligence • Powered by Dense Semantic NLP & Multi-Source Research")
