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

# Import Modular UI Components
from components.header import render_header
from components.sidebar import render_sidebar, SECTION_TO_NAV
from components.kpi_cards import render_kpi_strip
from components.topic_cards import render_whats_emerging
from components.charts import render_trend_velocity_chart
from components.monitoring import render_live_monitoring_feed
from components.alerts import render_alert_center_panel
from components.topic_details import render_topic_details_view
from components.pipeline import render_pipeline_loop

# -----------------------------------------------------------------------------
# Streamlit Configuration & Light Enterprise Styling Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrendRadar AI | Real-Time Emerging Topics Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External CSS Design System & Page Stylesheets
def load_design_system():
    page_css_files = [
        "trendradar.css",
        "overview.css",
        "monitoring.css",
        "topics.css",
        "alerts.css",
        "research.css",
        "workspace.css",
        "analytics.css",
        "research_lab.css"
    ]
    combined_css = []
    for fname in page_css_files:
        fpath = os.path.join(BASE_DIR, "styles", fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                combined_css.append(f.read())
    if combined_css:
        st.markdown(f"<style>{''.join(combined_css)}</style>", unsafe_allow_html=True)

load_design_system()

# -----------------------------------------------------------------------------
# Data Loaders (Robust Multilevel Production Loaders)
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    e6_alerts_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "emergence_alerts.csv")
    e6_trajs_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "daily_trajectories.csv")
    e6_posts_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "daily_clusters.csv")
    e6_base_p = os.path.join(BASE_DIR, "experiments", "e6_combined", "baseline_volume_alerts_daily.csv")
    
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
# Session State Defaults
# -----------------------------------------------------------------------------
if "nav_section" not in st.session_state:
    st.session_state["nav_section"] = "OVERVIEW"

default_tid = "TRAJ_284"
if not alerts_df.empty and "trajectory_id" in alerts_df.columns:
    if default_tid not in alerts_df["trajectory_id"].values:
        vetted = alerts_df[alerts_df.get("is_vetted_candidate", False) == True]
        default_tid = vetted["trajectory_id"].iloc[0] if not vetted.empty else alerts_df["trajectory_id"].iloc[0]

if "selected_trajectory_id" not in st.session_state:
    st.session_state["selected_trajectory_id"] = default_tid

if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("GEMINI_API_KEY", "")

if "source_filter" not in st.session_state:
    st.session_state["source_filter"] = "Reddit"

if "acknowledged_alerts" not in st.session_state:
    st.session_state["acknowledged_alerts"] = set()

if "watched_alerts" not in st.session_state:
    st.session_state["watched_alerts"] = set()

if "dismissed_alerts" not in st.session_state:
    st.session_state["dismissed_alerts"] = set()


def navigate_to(section_name: str, trajectory_id: str = None):
    st.session_state["nav_section"] = section_name
    target_nav = SECTION_TO_NAV.get(section_name, section_name)
    st.session_state["main_sidebar_nav_selector"] = target_nav
    st.session_state["_last_synced_section"] = section_name
    if trajectory_id:
        st.session_state["selected_trajectory_id"] = trajectory_id
    st.rerun()


# Helper to retrieve active topic record safely
def get_current_topic_row(tid: str):
    if alerts_df.empty or "trajectory_id" not in alerts_df.columns:
        return None
    matches = alerts_df[alerts_df["trajectory_id"] == tid]
    if not matches.empty:
        return matches.iloc[0]
    return alerts_df.iloc[0]


current_topic = get_current_topic_row(st.session_state["selected_trajectory_id"])

# -----------------------------------------------------------------------------
# RENDER SIDEBAR & TOP BAR
# -----------------------------------------------------------------------------
render_sidebar(current_topic)
render_header(st.session_state["source_filter"])

# Filter alerts_df if search query is active
filtered_alerts_df = alerts_df
search_q = st.session_state.get("search_query", "").strip().lower()
if search_q and not alerts_df.empty:
    filtered_alerts_df = alerts_df[
        alerts_df["topic_name"].astype(str).str.lower().str.contains(search_q) |
        alerts_df["trajectory_id"].astype(str).str.lower().str.contains(search_q) |
        alerts_df["representative_post"].astype(str).str.lower().str.contains(search_q)
    ]

# -----------------------------------------------------------------------------
# 1. OVERVIEW (Executive Intelligence Command Center)
# -----------------------------------------------------------------------------
if st.session_state["nav_section"] == "OVERVIEW":
    st.markdown('<div class="tr-section-title">EXECUTIVE INTELLIGENCE OVERVIEW</div>', unsafe_allow_html=True)

    # 1. KPI Strip (6 Cards)
    render_kpi_strip(alerts_df, trajs_df, posts_df, base_df)

    # 2. What's Emerging Right Now (3x2 Grid with Rapidly Growing Topics chart)
    render_whats_emerging(alerts_df, navigate_to)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Middle Split: Trend Velocity & Live Monitoring
    c_vel, c_mon = st.columns([3, 2])
    with c_vel:
        render_trend_velocity_chart(alerts_df, st.session_state["selected_trajectory_id"])
    with c_mon:
        render_live_monitoring_feed(posts_df)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 4. Lower Split: Emerging Topics Table Registry & Alert Center
    c_tab, c_alt = st.columns([3, 2])
    with c_tab:
        st.markdown('<div class="tr-section-title">EMERGING TOPICS REGISTRY (SELECTABLE)</div>', unsafe_allow_html=True)
        cols_show = ["trajectory_id", "lifecycle_status", "total_volume", "avg_coherence", "emergence_score", "birth_window"]
        avail_cols = [c for c in cols_show if c in filtered_alerts_df.columns]
        if not filtered_alerts_df.empty:
            st.dataframe(filtered_alerts_df[avail_cols].head(10), width="stretch", hide_index=True)
        else:
            st.info("No topics matching search query.")

    with c_alt:
        render_alert_center_panel(alerts_df, navigate_to)


# -----------------------------------------------------------------------------
# 2. MONITORING (Live Ingestion & Stream Console)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "MONITORING":
    st.markdown('<div class="tr-section-title">LIVE MONITORING CONSOLE</div>', unsafe_allow_html=True)
    st.write("Real-time chronological stream filtered for boilerplate templates, encoded via Sentence-BERT, and mapped into cluster centroids.")

    m_col1, m_col2, m_col3 = st.columns([1, 1, 2])
    with m_col1:
        st.selectbox("Channel Source", ["All Sources", "Reddit r/AskReddit", "Twitter/X", "Forums"])
    with m_col2:
        st.slider("Semantic Similarity Floor", 0.50, 0.99, 0.68)
    with m_col3:
        st.text_input("Filter Stream Keyword", placeholder="Search live posts...")

    st.markdown("### Incoming Chronological Post Stream")
    if not posts_df.empty:
        disp_posts = posts_df[["window_id", "cluster_id", "clean_text"]].head(25) if "clean_text" in posts_df.columns else posts_df.head(25)
        st.dataframe(disp_posts, width="stretch", hide_index=True)
    else:
        st.info("Live stream buffering incoming posts.")


# -----------------------------------------------------------------------------
# 2b. EMERGING TOPICS (Registry Directory)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "EMERGING TOPICS":
    st.markdown('<div class="tr-section-title">EMERGING TOPICS REGISTRY</div>', unsafe_allow_html=True)
    st.write("Complete directory of detected topic trajectories, emergence scores, coherence metrics, and lifecycle stages.")
    cols_show = ["trajectory_id", "lifecycle_status", "total_volume", "avg_coherence", "emergence_score", "birth_window", "representative_post"]
    avail_cols = [c for c in cols_show if c in filtered_alerts_df.columns]
    if not filtered_alerts_df.empty:
        st.dataframe(filtered_alerts_df[avail_cols], width="stretch", hide_index=True)
    else:
        st.info("No emerging topics matching search filter.")


# -----------------------------------------------------------------------------
# 3. TOPICS (Topic Details / Deep Dive & Explainable AI)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "TOPICS":
    all_tids = alerts_df["trajectory_id"].tolist() if not alerts_df.empty else []
    render_topic_details_view(current_topic, all_tids, navigate_to)


# -----------------------------------------------------------------------------
# 4. ALERTS (Dedicated Alert Center)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "ALERTS":
    st.markdown('<div class="tr-section-title">PRIORITIZED ALERT CENTER</div>', unsafe_allow_html=True)
    st.write("Action high-velocity topics or place nascent seeds on watchlist.")

    bursts = alerts_df[alerts_df.get("lifecycle_status", "") == "[BURST]"]
    confirmed = alerts_df[alerts_df.get("lifecycle_status", "") == "[CONFIRMED_SEED]"]

    st.markdown(f"### High-Priority Alerts ({len(bursts)} Bursts • {len(confirmed)} Confirmed Seeds)")
    for _, r in bursts.head(8).iterrows():
        tid = r["trajectory_id"]
        t_name = str(r["topic_name"]).replace("[", "").replace("]", "").replace("'", "")
        st.markdown(f"""
        <div class="tr-alert-banner">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#B91C1C;">🚨 #{tid} — {t_name}</h4>
                <span class="tr-badge tr-badge-emerging">BURST • SCORE {r.get('emergence_score', 0.0):.3f}</span>
            </div>
            <p style="margin:6px 0; color:#334155; font-size:0.86rem;">"{r.get('representative_post', '')}"</p>
            <div style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono', monospace;">
                Volume Path: <code>{r.get('volume_path')}</code> &nbsp;|&nbsp; Coherence: <strong>{r.get('avg_coherence', 0.0):.3f}</strong> &nbsp;|&nbsp; Detected: <strong>{r.get('birth_window')}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        c_a1, c_a2, c_a3 = st.columns(3)
        with c_a1:
            if st.button("INVESTIGATE", key=f"al_inv_{tid}", use_container_width=True):
                navigate_to("TOPICS", tid)
        with c_a2:
            if st.button("RESEARCH CONTEXT", key=f"al_res_{tid}", use_container_width=True):
                navigate_to("RESEARCH", tid)
        with c_a3:
            if st.button("PIN CASE", key=f"al_pin_{tid}", use_container_width=True):
                save_investigation(tid, t_name, "[BURST]", str(r.get("volume_path")), float(r.get("avg_coherence", 0.0)), str(r.get("birth_window")), str(r.get("representative_post")), [k.strip() for k in t_name.split(",")])
                st.toast(f"Pinned {tid} to Desk", icon="📌")


# -----------------------------------------------------------------------------
# 5. RESEARCH (Research Hub)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "RESEARCH":
    st.markdown('<div class="tr-section-title">RESEARCH HUB & EXTERNAL CORROBORATION</div>', unsafe_allow_html=True)
    st.write("Separates internal platform evidence from external knowledge retrieval (**Wikipedia**, **arXiv**).")

    t_row = current_topic
    if t_row is not None:
        tid = t_row["trajectory_id"]
        t_name = str(t_row["topic_name"]).replace("[", "").replace("]", "").replace("'", "")
        keywords = [k.strip() for k in t_name.split(",")]
        rep_post = str(t_row.get("representative_post", ""))

        st.markdown(f"### Research Dossier: #{tid} — {t_name}")

        with st.spinner("Querying live Wikipedia & arXiv APIs..."):
            dossier = get_topic_research_dossier(t_name, keywords, rep_post)

        col_ev_left, col_ev_right = st.columns(2)
        with col_ev_left:
            st.markdown("""
            <div style="background:#FFFFFF; border:1px solid var(--tr-border); border-radius:10px; padding:18px; margin-bottom:14px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
                <h4 style="margin:0 0 8px 0; color:#0F172A;">📊 Internal Platform Evidence (TrendRadar AI)</h4>
            """, unsafe_allow_html=True)
            st.write(f"- **Volume Path**: `{t_row.get('volume_path')}`")
            st.write(f"- **Cluster Coherence**: `{t_row.get('avg_coherence', 0.0):.3f}`")
            st.write(f"- **Detection Window**: `{t_row.get('birth_window')}`")
            st.write(f"- **Extracted Entities**: `{t_row.get('entities', 'None')}`")
            st.markdown(f"**Canonical Post:** *\"{rep_post}\"*")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ev_right:
            st.markdown("""
            <div style="background:#FFFFFF; border:1px solid var(--tr-border); border-radius:10px; padding:18px; margin-bottom:14px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
                <h4 style="margin:0 0 8px 0; color:#0F172A;">🌐 External Context Corroboration</h4>
            """, unsafe_allow_html=True)
            conf = dossier.get("corroboration_confidence", "Verified External Topic")
            st.markdown(f"**Corroboration Rating:** <span class='tr-badge tr-badge-confirmed'>{conf}</span>", unsafe_allow_html=True)
            wiki = dossier.get("wikipedia", {})
            if wiki.get("found"):
                st.write(f"**Wikipedia Title:** [{wiki.get('title')}]({wiki.get('url')})")
                st.write(wiki.get("summary"))
            else:
                st.info("No direct Wikipedia entry found. Community discourse is platform-specific.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🧠 Synthesize AI Briefing with this Evidence Package", use_container_width=True):
            st.session_state["cached_dossier"] = dossier
            navigate_to("AI BRIEFING", tid)


# -----------------------------------------------------------------------------
# 6. AI BRIEFING (AI Executive Briefing Engine)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "AI BRIEFING":
    st.markdown('<div class="tr-section-title">AI EXECUTIVE BRIEFING</div>', unsafe_allow_html=True)
    st.write("Explains the mathematically proven evidence without hallucinating non-existent trends.")

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

        with st.spinner("Generating executive intelligence brief..."):
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

        st.markdown(f"### Executive Intelligence Brief: #{tid} — {t_name}")

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("""
            <div class="tr-brief-card what">
                <div class="tr-brief-title" style="color:#2563EB;">📌 1. What is Happening?</div>
            """, unsafe_allow_html=True)
            st.write(brief.get("what_is_happening"))
            st.markdown("</div>", unsafe_allow_html=True)

        with b2:
            st.markdown("""
            <div class="tr-brief-card why">
                <div class="tr-brief-title" style="color:#D97706;">🚀 2. Why is it Emerging?</div>
            """, unsafe_allow_html=True)
            st.write(brief.get("why_is_it_emerging"))
            st.markdown("</div>", unsafe_allow_html=True)

        b3, b4 = st.columns(2)
        with b3:
            st.markdown("""
            <div class="tr-brief-card evidence">
                <div class="tr-brief-title" style="color:#059669;">📊 3. Supporting Evidence & Metrics</div>
            """, unsafe_allow_html=True)
            st.write(brief.get("evidence_summary"))
            st.markdown("</div>", unsafe_allow_html=True)

        with b4:
            st.markdown("""
            <div class="tr-brief-card next">
                <div class="tr-brief-title" style="color:#7C3AED;">🎯 4. Recommended Next Steps</div>
            """, unsafe_allow_html=True)
            st.write(brief.get("recommended_next_steps"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("⚡ Open Action Hub to Dispatch Alert", use_container_width=True):
            navigate_to("ACTION HUB", tid)


# -----------------------------------------------------------------------------
# 7. ACTION HUB (Dispatch & Reporting)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "ACTION HUB":
    st.markdown('<div class="tr-section-title">ACTION HUB & DISPATCH</div>', unsafe_allow_html=True)
    st.write("Convert intelligence into formal briefs, stakeholder alerts, and committed workspace cases.")

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

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("### 📄 Formal Intelligence Brief")
            brief_md = brief.get("full_brief_markdown", "No content.")
            st.download_button(
                label="📥 Download 1-Page Intelligence Brief (.md)",
                data=brief_md,
                file_name=f"TrendRadar_Brief_{tid}.md",
                mime="text/markdown",
                use_container_width=True
            )

            st.markdown("### 📢 Stakeholder Alert (Slack / Teams Ready)")
            slack_alert = (
                f"🚨 *TRENDRADAR AI ALERT: {t_name.upper()}*\n"
                f"• *Status*: {status} | *First Detected*: {birth} | *Volume Path*: `{vol_path}`\n"
                f"• *Core Theme*: {brief.get('what_is_happening')[:180]}...\n"
                f"• *Recommended Action*: {brief.get('recommended_next_steps')[:140]}...\n"
                f"• *Analyst Dashboard*: http://localhost:8501"
            )
            st.code(slack_alert, language="markdown")

        with col_a2:
            st.markdown("### 💾 Commit Investigation to Workspace")
            with st.form("workspace_form_act"):
                c_status = st.selectbox("Investigation Status", ["Active", "Monitoring", "Escalated", "Resolved"])
                c_notes = st.text_area("Analyst Notes & Strategic Context", "High-velocity signal verified. Monitor next checkpoint.")
                commit_btn = st.form_submit_button("Commit Case to Workspace", use_container_width=True)
                if commit_btn:
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
                        analyst_notes=c_notes,
                        status=c_status
                    )
                    st.success(f"Case `{tid}` committed to Workspace with status '{c_status}'!")


# -----------------------------------------------------------------------------
# 8. WORKSPACE (Investigation Desk)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "WORKSPACE":
    st.markdown('<div class="tr-section-title">WORKSPACE & INVESTIGATION DESK</div>', unsafe_allow_html=True)
    st.write("Track ongoing topic investigations, update statuses, and review pinned intelligence briefs.")

    investigations = list_investigations()

    if not investigations:
        st.info("No investigations currently logged. Go to **Overview** or **Topics** to pin a topic.")
    else:
        k_w1, k_w2, k_w3 = st.columns(3)
        k_w1.metric("Total Monitored Cases", len(investigations))
        active_cnt = len([i for i in investigations if i.get("status") in ["Active", "Escalated"]])
        k_w2.metric("Active / Escalated", active_cnt)
        mon_cnt = len([i for i in investigations if i.get("status") == "Monitoring"])
        k_w3.metric("Under Monitoring", mon_cnt)

        st.download_button(
            "📥 Export Master Workspace Dossier (.md)",
            data=export_workspace_markdown(),
            file_name="TrendRadar_Master_Workspace.md",
            mime="text/markdown"
        )

        st.markdown("---")
        st.markdown("### Saved Investigation Cases")
        for inv in investigations:
            inv_id = inv["id"]
            with st.expander(f"[{inv.get('status', 'Active')}] {inv['topic_name']} — {inv['trajectory_id']}"):
                ci1, ci2 = st.columns([2, 1])
                with ci1:
                    st.write(f"**Canonical Post:** *\"{inv.get('representative_post')}\"*")
                    st.write(f"**Volume Path:** `{inv.get('volume_path')}` | **Coherence:** `{inv.get('avg_coherence')}`")
                    st.write(f"**Analyst Notes:** {inv.get('analyst_notes')}")
                    st.caption(f"Last Updated: {inv.get('updated_at')}")

                with ci2:
                    new_st = st.selectbox(
                        "Update Status",
                        ["Active", "Monitoring", "Escalated", "Resolved"],
                        index=["Active", "Monitoring", "Escalated", "Resolved"].index(inv.get("status", "Active")),
                        key=f"st_sel_{inv_id}"
                    )
                    new_nt = st.text_input("Edit Notes", value=inv.get("analyst_notes", ""), key=f"nt_sel_{inv_id}")
                    if st.button("Save Update", key=f"btn_sv_{inv_id}"):
                        update_investigation_status(inv_id, new_st, new_nt)
                        st.toast(f"Updated {inv_id}", icon="✅")
                        st.rerun()

                    if st.button("🗑️ Delete Case", key=f"btn_dl_{inv_id}"):
                        delete_investigation(inv_id)
                        st.toast(f"Deleted {inv_id}", icon="🗑️")
                        st.rerun()


# -----------------------------------------------------------------------------
# 9. ANALYTICS (Comparative Evaluation & Benchmark)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "ANALYTICS":
    st.markdown('<div class="tr-section-title">ANALYTICS & BENCHMARK AUDITS</div>', unsafe_allow_html=True)
    st.write("Head-to-head empirical comparison: Keyword Frequency Monitoring vs. TrendRadar AI Dense Semantic Detection.")

    c_an1, c_an2 = st.columns([1, 1])
    with c_an1:
        comp_df = pd.DataFrame({
            "Method": ["Keyword Baseline", "TrendRadar AI Pipeline"],
            "Alert Count": [823, 102],
            "Type": ["Fragmented Term Alerts", "Coherent Topic Narratives"]
        })
        fig_bar = px.bar(
            comp_df,
            x="Method",
            y="Alert Count",
            color="Method",
            text="Alert Count",
            title="Total Alert Triggers: Term-Level vs. Topic-Level",
            color_discrete_sequence=["#DC2626", "#059669"]
        )
        fig_bar.update_layout(
            height=340,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            showlegend=False,
            font=dict(color="#475569")
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B"))
        fig_bar.update_yaxes(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B"))
        st.plotly_chart(fig_bar, width="stretch")

    with c_an2:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid var(--tr-border); border-radius:10px; padding:22px; margin-top:20px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
            <div style="font-size:0.85rem; color:#059669; font-weight:700; text-transform:uppercase;">SCIENTIFICALLY DEFENSIBLE BENCHMARK</div>
            <div style="font-size:2.8rem; font-weight:900; color:#0F172A; margin:8px 0; font-family:'JetBrains Mono', monospace;">87.6%</div>
            <div style="font-size:0.95rem; color:#334155; font-weight:600;">Alert Noise Reduction</div>
            <p style="font-size:0.84rem; color:#64748B; margin-top:8px;">
                Traditional keyword-frequency monitoring triggered 823 disconnected spikes on common conversational terms ("help", "money", "people"). TrendRadar AI grouped the entire corpus into 102 clean, actionable topic events.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 10. RESEARCH LAB (Ablations & Scientific Progression)
# -----------------------------------------------------------------------------
elif st.session_state["nav_section"] == "RESEARCH LAB":
    st.markdown('<div class="tr-section-title">SCIENTIFIC RESEARCH LAB</div>', unsafe_allow_html=True)
    st.write("Rigorous empirical evaluation suite: Controlled Ablation progression (E0 to E6), lead-time evaluation, and observable density boundary.")

    lab_t1, lab_t2 = st.tabs(["Master Ablation Progression (E0–E6)", "Lead-Time Evaluation & Density Boundaries"])

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
        ### Key Scientific Findings
        - **E0 -> E1 (Scale Normalization):** Solved sample-size mismatch, allowing OnlyFans (11-12 posts) to be detected.
        - **E1 -> E2 (Template Stripping):** Suppressed **82.4% (28/34)** of conversational question-template bursts while surfacing Hurricane Ida.
        - **E2 -> E3 (Lexical Consistency Gate):** Eliminated the 24-checkpoint semantic drift pseudo-chain (`TRAJ_002`), capping max lifespan at 5 checkpoints.
        - **E5 -> E6 (Two-Tier Architecture):** Mitigated the sensitivity-inflation trade-off by decoupling standard topic detection ($N \ge 7$) from quarantined incubation ($3 \le N < 7$).
        """)

    with lab_t2:
        st.subheader("Checkpoint Lead-Time & Observable Density Boundary")
        if not eval_df.empty:
            st.dataframe(eval_df, width="stretch", hide_index=True)

        st.markdown(r"""
        ### Observable Detection Boundary
        - **Single-Checkpoint Clustering Limit:** HDBSCAN requires at least $N = 3$ co-occurring posts to form a cluster core.
        - Events with $\le 2$ posts per day (e.g. Reddit locked comments bug) are mathematically indistinguishable from noise within a single time slice.
        - **Strategic Extension:** Continuous 48–72h sliding vector buffers to accumulate sparse signals across time slices.
        """)


# -----------------------------------------------------------------------------
# BOTTOM TECHNICAL PIPELINE PROCESS LOOP
# -----------------------------------------------------------------------------
render_pipeline_loop()
