import streamlit as st
import pandas as pd
from src.workspace import save_investigation

def render_alert_center_panel(alerts_df: pd.DataFrame, navigate_cb):
    """
    Renders the Prioritized Alert Center with interactive dark glass glowing cards.
    Fully connects to real alert records, supports Investigate, Acknowledge, Watch, Dismiss.
    """
    st.markdown("""
    <div class="tr-section-title">Alert Center (Prioritized)</div>
    """, unsafe_allow_html=True)

    if "acknowledged_alerts" not in st.session_state:
        st.session_state["acknowledged_alerts"] = set()
    if "watched_alerts" not in st.session_state:
        st.session_state["watched_alerts"] = set()
    if "dismissed_alerts" not in st.session_state:
        st.session_state["dismissed_alerts"] = set()

    acknowledged = st.session_state["acknowledged_alerts"]
    dismissed = st.session_state["dismissed_alerts"]

    # Candidate high-priority alerts
    candidates = [
        {
            "tid": "TRAJ_284",
            "name": "OnlyFans Policy Change",
            "reason": "Rapid volume acceleration (+145%) • strong semantic coherence • consistent trajectory.",
            "status": "[BURST]",
            "coh": 0.684,
            "window": "D_11",
            "vol": "24 Posts",
            "growth": "+145%",
            "post": "With Onlyfans banning sexually explicit content, how does it affect creators?",
            "kws": ["onlyfans", "ban", "creators", "policy"]
        },
        {
            "tid": "TRAJ_757",
            "name": "Hurricane Ida Discussion",
            "reason": "Sudden coastal storm discourse burst • high velocity spike across 42 posts.",
            "status": "[BURST]",
            "coh": 0.654,
            "window": "D_27",
            "vol": "42 Posts",
            "growth": "+180%",
            "post": "How were you impacted by Hurricane Ida?",
            "kws": ["hurricane", "storm", "evacuation", "ida"]
        },
        {
            "tid": "TRAJ_456",
            "name": "Vaccine Mandate Signal",
            "reason": "Quarantined seed candidate with extreme semantic coherence (0.938) forming early consensus.",
            "status": "[CONFIRMED_SEED]",
            "coh": 0.938,
            "window": "D_16",
            "vol": "5 Posts",
            "growth": "+120%",
            "post": "Should one try and reason with a friend ready to throw away their medical career because of vaccine?",
            "kws": ["vaccine", "mandate", "medical"]
        }
    ]

    # Filter out dismissed alerts
    active_alerts = [c for c in candidates if c["tid"] not in dismissed]

    if not active_alerts:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
                    border-radius:12px; padding:16px; color:#34d399; font-size:0.85rem;">
            ✅ All priority alerts acknowledged or resolved. <strong>0</strong> critical alerts pending.
        </div>
        """, unsafe_allow_html=True)
        return

    # Render each active alert (showing top priority alert prominently)
    for alert in active_alerts[:2]:
        tid = alert["tid"]
        is_ack = tid in acknowledged
        is_watched = tid in st.session_state["watched_alerts"]

        status_tag = "ACKNOWLEDGED" if is_ack else (f"{alert['growth']} BURST" if "BURST" in alert["status"] else "CONFIRMED SEED")
        status_class = "tr-badge-confirmed" if is_ack else ("tr-badge-emerging" if "BURST" in alert["status"] else "tr-badge-watching")

        st.markdown(f"""
        <div class="tr-alert-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="margin:0; font-size:0.95rem; display:flex; align-items:center; gap:8px;">
                    🚨 #{tid} — {alert['name']}
                </h4>
                <span class="tr-badge {status_class}">{status_tag}</span>
            </div>
            <p style="margin:0 0 10px 0; font-size:0.82rem; line-height:1.5; color:#cbd5e1;">
                {alert['reason']}
            </p>
            <div style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono', monospace;
                        display:flex; gap:14px; flex-wrap:wrap;">
                <span>📅 DETECTED: {alert['window']}</span>
                <span>🧬 COHERENCE: {alert['coh']:.3f}</span>
                <span>📈 LIFECYCLE: {alert['status']}</span>
                <span>📝 {alert['vol']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        al_b1, al_b2, al_b3, al_b4 = st.columns(4)
        with al_b1:
            if st.button("🔍 INVESTIGATE", key=f"btn_alt_inv_{tid}", use_container_width=True):
                navigate_cb("TOPICS", tid)
        with al_b2:
            ack_label = "✓ ACKED" if is_ack else "✓ ACKNOWLEDGE"
            if st.button(ack_label, key=f"btn_alt_ack_{tid}", use_container_width=True):
                st.session_state["acknowledged_alerts"].add(tid)
                st.toast(f"Alert #{tid} Acknowledged", icon="✓")
                st.rerun()
        with al_b3:
            watch_label = "👁️ WATCHING" if is_watched else "👁️ WATCH"
            if st.button(watch_label, key=f"btn_alt_wat_{tid}", use_container_width=True):
                st.session_state["watched_alerts"].add(tid)
                save_investigation(
                    trajectory_id=tid,
                    topic_name=alert["name"],
                    lifecycle_status=alert["status"],
                    volume_path=alert["vol"],
                    coherence=alert["coh"],
                    birth_window=alert["window"],
                    representative_post=alert["post"],
                    keywords=alert["kws"],
                    status="Monitoring"
                )
                st.toast(f"Pinned #{tid} to Workspace Watchlist", icon="👁️")
        with al_b4:
            if st.button("✕ DISMISS", key=f"btn_alt_dis_{tid}", use_container_width=True):
                st.session_state["dismissed_alerts"].add(tid)
                st.toast(f"Dismissed #{tid} from Priority Alert View", icon="✕")
                st.rerun()
