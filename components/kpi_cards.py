import streamlit as st
import pandas as pd

def render_kpi_strip(alerts_df: pd.DataFrame, trajs_df: pd.DataFrame, posts_df: pd.DataFrame, base_df: pd.DataFrame):
    """
    Renders 6 KPI cards with dark glassmorphism + gradient accent design
    reflecting the held-out evaluation benchmark (D_21–D_30).
    """
    total_posts = len(posts_df) if not posts_df.empty else 8784

    # Held-out benchmark metrics (D_21 to D_30)
    baseline_alerts_count = 171
    pipeline_candidates_count = 39
    novelty_gated_alerts_count = 14
    quarantined_chatter_count = 25
    real_events_detected = "2 / 5"
    gated_precision = "0.1429"

    st.markdown(f"""
    <div class="tr-kpi-container">
        <div class="tr-kpi-card">
            <div class="tr-kpi-label">🔍 Pipeline Candidates</div>
            <div class="tr-kpi-val">{pipeline_candidates_count}</div>
            <div class="tr-kpi-delta dim">Held-Out D21–D30</div>
        </div>
        <div class="tr-kpi-card">
            <div class="tr-kpi-label">🛡️ Novelty Gate Alerts</div>
            <div class="tr-kpi-val">{novelty_gated_alerts_count}</div>
            <div class="tr-kpi-delta pos">↓ {quarantined_chatter_count} Quarantined</div>
        </div>
        <div class="tr-kpi-card">
            <div class="tr-kpi-label">🎯 Detected Events</div>
            <div class="tr-kpi-val">{real_events_detected}</div>
            <div class="tr-kpi-delta crit">Strict Recall 40.0%</div>
        </div>
        <div class="tr-kpi-card">
            <div class="tr-kpi-label">🚨 Keyword Baseline</div>
            <div class="tr-kpi-val">{baseline_alerts_count}</div>
            <div class="tr-kpi-delta warn">Keyword Spikes</div>
        </div>
        <div class="tr-kpi-card">
            <div class="tr-kpi-label">📊 Posts Monitored</div>
            <div class="tr-kpi-val">{total_posts:,}</div>
            <div class="tr-kpi-delta dim">30 Chrono Slices</div>
        </div>
        <div class="tr-kpi-card">
            <div class="tr-kpi-label">✅ Gated Precision</div>
            <div class="tr-kpi-val">{gated_precision}</div>
            <div class="tr-kpi-delta pos">vs 0.0175 Baseline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
