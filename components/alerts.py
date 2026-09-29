import streamlit as st
import pandas as pd
from src.workspace import save_investigation

def render_alert_center_panel(alerts_df: pd.DataFrame, navigate_cb):
    """
    Renders the dedicated Alert Center panel matching Section 12.
    Interactive actions: INVESTIGATE, ACKNOWLEDGE, WATCH, DISMISS.
    """
    st.markdown('<div class="tr-section-title">ALERT CENTER (PRIORITIZED)</div>', unsafe_allow_html=True)

    target_tid = "TRAJ_284"
    is_dismissed = target_tid in st.session_state.get("dismissed_alerts", set())
    is_ack = target_tid in st.session_state.get("acknowledged_alerts", set())

    if not is_dismissed:
        status_tag = "ACKNOWLEDGED" if is_ack else "+145% BURST"
        status_class = "tr-badge-confirmed" if is_ack else "tr-badge-emerging"

        st.markdown(f"""
        <div class="tr-alert-banner">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#B91C1C; font-size:1.02rem;">🚨 NEW EMERGING TOPIC: OnlyFans Policy Change</h4>
                <span class="tr-badge {status_class}">{status_tag}</span>
            </div>
            <p style="margin:6px 0; font-size:0.83rem; color:#334155;">
                Rapid growth • strong semantic coherence (0.684) • consistent trajectory across consecutive checkpoints.
            </p>
            <div style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono', monospace;">
                DETECTED: D_11 &nbsp;|&nbsp; COHERENCE: 0.684 &nbsp;|&nbsp; LIFECYCLE: [BURST] &nbsp;|&nbsp; 24 Posts
            </div>
        </div>
        """, unsafe_allow_html=True)

        al_b1, al_b2, al_b3, al_b4 = st.columns(4)
        with al_b1:
            if st.button("INVESTIGATE", key="btn_alert_inv", use_container_width=True):
                navigate_cb("TOPICS", target_tid)
        with al_b2:
            if st.button("ACKNOWLEDGE", key="btn_alert_ack", use_container_width=True):
                st.session_state["acknowledged_alerts"].add(target_tid)
                st.toast("Alert Acknowledged by Analyst", icon="✓")
                st.rerun()
        with al_b3:
            if st.button("WATCH", key="btn_alert_wat", use_container_width=True):
                st.session_state["watched_alerts"].add(target_tid)
                save_investigation(
                    trajectory_id=target_tid,
                    topic_name="OnlyFans Policy Change",
                    lifecycle_status="[BURST]",
                    volume_path="11 -> 24",
                    coherence=0.684,
                    birth_window="D_11",
                    representative_post="With Onlyfans banning sexually explicit content, how does it affect creators?",
                    keywords=["onlyfans", "ban", "explicit", "creators"],
                    status="Monitoring"
                )
                st.toast("Saved OnlyFans to Workspace Watchlist", icon="👁️")
        with al_b4:
            if st.button("DISMISS", key="btn_alert_dis", use_container_width=True):
                st.session_state["dismissed_alerts"].add(target_tid)
                st.toast("Alert Dismissed from Active Priority", icon="✕")
                st.rerun()
    else:
        st.info("Priority alert for TRAJ_284 dismissed. 0 active critical alerts pending.")
