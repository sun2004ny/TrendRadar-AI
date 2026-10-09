import streamlit as st
import pandas as pd
from src.workspace import save_investigation
from src.alerts_db import get_alerts_db

def _ensure_initial_seed_alerts(db):
    """Populates seed alerts into the database if empty, prioritizing novelty-gated alerts."""
    counts = db.count_by_status()
    total_alerts = sum(counts.values())
    if total_alerts == 0:
        seed_data = [
            {
                "alert_id": "ALT_TRAJ_757",
                "trajectory_id": "TRAJ_757",
                "checkpoint_timestamp": "D_27",
                "day": "D_27",
                "title": "Hurricane Ida Landfall (Held-Out Real Event)",
                "keywords": ["hurricane", "storm", "evacuation", "ida"],
                "post_count": 42,
                "coherence": 0.654,
                "velocity": 2.80,
                "acceleration": 1.15,
                "emergence_score": 0.95,
                "novelty_score": 0.35,
                "sample_posts": ["How were you impacted by Hurricane Ida?"],
                "note": "Passed Causal Novelty Gate (sim: 0.35 < 0.70) • Verified breaking disaster on held-out D_27."
            },
            {
                "alert_id": "ALT_SEED_579",
                "trajectory_id": "SEED_579",
                "checkpoint_timestamp": "D_26",
                "day": "D_26",
                "title": "Reddit iOS Comment Lock Bug (Held-Out Real Event)",
                "keywords": ["reddit", "ios", "locked", "bug", "comments"],
                "post_count": 4,
                "coherence": 0.8996,
                "velocity": 2.10,
                "acceleration": 0.90,
                "emergence_score": 0.94,
                "novelty_score": 0.42,
                "sample_posts": ["Why are all the OP comments locked on iOS Reddit app?"],
                "note": "Passed Causal Novelty Gate (sim: 0.42 < 0.70) • Promoted sparse seed on held-out D_26."
            },
            {
                "alert_id": "ALT_TRAJ_284",
                "trajectory_id": "TRAJ_284",
                "checkpoint_timestamp": "D_11",
                "day": "D_11",
                "title": "OnlyFans Policy Change (Tuning Real Event)",
                "keywords": ["onlyfans", "ban", "creators", "policy"],
                "post_count": 24,
                "coherence": 0.684,
                "velocity": 2.45,
                "acceleration": 0.82,
                "emergence_score": 0.92,
                "novelty_score": 0.58,
                "sample_posts": ["With Onlyfans banning sexually explicit content, how does it affect creators?"],
                "note": "Passed Causal Novelty Gate (sim: 0.58 < 0.70) • Verified breaking policy event on D_11."
            },
            {
                "alert_id": "ALT_TRAJ_456",
                "trajectory_id": "TRAJ_456",
                "checkpoint_timestamp": "D_16",
                "day": "D_16",
                "title": "Vaccine Mandate Signal",
                "keywords": ["vaccine", "mandate", "medical"],
                "post_count": 5,
                "coherence": 0.938,
                "velocity": 1.20,
                "acceleration": 0.40,
                "emergence_score": 0.88,
                "novelty_score": 0.60,
                "sample_posts": ["Should one try and reason with a friend ready to throw away their medical career because of vaccine?"],
                "note": "Quarantined seed candidate with extreme semantic coherence (0.938) forming early consensus."
            }
        ]
        for item in seed_data:
            db.create_or_get_alert(item)


def render_alert_center_panel(alerts_df: pd.DataFrame, navigate_cb):
    """
    Renders the Prioritized Alert Center backed by SQLite AlertsDB in WAL mode.
    Prioritizes Causal Novelty-Gated alerts (similarity < 0.70) at the top of the feed.
    """
    st.markdown("""
    <div class="tr-section-title">Alert Center (Lifecycle Managed — Novelty Gated First)</div>
    """, unsafe_allow_html=True)

    db = get_alerts_db()
    _ensure_initial_seed_alerts(db)

    # Query active alerts and sort so novelty-gated alerts (novelty_score < 0.70) appear first
    active_alerts = db.get_active_alerts(limit=20)
    active_alerts = sorted(
        active_alerts,
        key=lambda a: (
            0 if float(a.get("novelty_score", 0.0)) < 0.70 else 1,
            -float(a.get("emergence_score", 0.0))
        )
    )
    counts = db.count_by_status()

    # Lifecycle Summary Strip
    st.markdown(f"""
    <div style="display:flex; gap:12px; margin-bottom:12px; font-size:0.75rem; font-family:'JetBrains Mono', monospace;">
        <span style="color:#f43f5e;">🚨 NEW: {counts.get('NEW', 0)}</span>
        <span style="color:#fbbf24;">✓ ACKNOWLEDGED: {counts.get('ACKNOWLEDGED', 0)}</span>
        <span style="color:#818cf8;">👁️ WATCHING: {counts.get('WATCHING', 0)}</span>
        <span style="color:#34d399;">✅ RESOLVED: {counts.get('RESOLVED', 0)}</span>
        <span style="color:#64748b;">✕ DISMISSED: {counts.get('DISMISSED', 0)}</span>
    </div>
    """, unsafe_allow_html=True)

    if not active_alerts:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
                    border-radius:12px; padding:16px; color:#34d399; font-size:0.85rem;">
            ✅ All priority alerts acknowledged or resolved. <strong>0</strong> active alerts pending.
        </div>
        """, unsafe_allow_html=True)
        return

    for alert in active_alerts[:3]:
        aid = alert["alert_id"]
        tid = alert["trajectory_id"]
        status = alert["status"]

        status_class = {
            "NEW": "tr-badge-emerging",
            "ACKNOWLEDGED": "tr-badge-confirmed",
            "WATCHING": "tr-badge-watching",
            "RESOLVED": "tr-badge-confirmed",
            "DISMISSED": "tr-badge-watching"
        }.get(status, "tr-badge-emerging")

        sample_post = alert["sample_posts"][0] if alert["sample_posts"] else ""
        is_novelty_passed = float(alert.get("novelty_score", 0.0)) < 0.70
        novelty_badge = '<span class="tr-badge tr-badge-confirmed" style="font-size:0.55rem; padding:2px 6px; margin-right:6px;">🛡️ NOVELTY PASSED</span>' if is_novelty_passed else ''

        st.markdown(f"""
        <div class="tr-alert-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="margin:0; font-size:0.95rem; display:flex; align-items:center; gap:8px;">
                    🚨 #{tid} — {alert['title']}
                </h4>
                <div>
                    {novelty_badge}
                    <span class="tr-badge {status_class}">{status}</span>
                </div>
            </div>
            <p style="margin:0 0 10px 0; font-size:0.82rem; line-height:1.5; color:#cbd5e1;">
                <strong>Keywords:</strong> {alert['keywords']}<br>
                <em>"{sample_post}"</em>
            </p>
            <div style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono', monospace;
                        display:flex; gap:14px; flex-wrap:wrap;">
                <span>📅 DETECTED: {alert['day']}</span>
                <span>🧬 COHERENCE: {alert['coherence']:.3f}</span>
                <span>📈 VELOCITY: {alert['velocity']:.2f}x</span>
                <span>📊 SCORE: {alert['emergence_score']:.2f}</span>
                <span>📝 {alert['post_count']} Posts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        al_b1, al_b2, al_b3, al_b4 = st.columns(4)
        with al_b1:
            if st.button("🔍 INVESTIGATE", key=f"btn_alt_inv_{aid}", use_container_width=True):
                navigate_cb("TOPICS", tid)

        with al_b2:
            if status == "NEW":
                if st.button("✓ ACKNOWLEDGE", key=f"btn_alt_ack_{aid}", use_container_width=True):
                    db.update_status(aid, "ACKNOWLEDGED", note="Operator acknowledged alert")
                    st.toast(f"Alert #{tid} Acknowledged", icon="✓")
                    st.rerun()
            elif status == "ACKNOWLEDGED":
                if st.button("✅ RESOLVE", key=f"btn_alt_res_{aid}", use_container_width=True):
                    db.update_status(aid, "RESOLVED", note="Operator resolved topic alert")
                    st.toast(f"Alert #{tid} Resolved", icon="✅")
                    st.rerun()
            else:
                st.button("✓ ACKED", key=f"btn_alt_acked_{aid}", disabled=True, use_container_width=True)

        with al_b3:
            if status == "WATCHING":
                st.button("👁️ WATCHING", key=f"btn_alt_wtd_{aid}", disabled=True, use_container_width=True)
            else:
                if st.button("👁️ WATCH", key=f"btn_alt_wat_{aid}", use_container_width=True):
                    if status == "NEW":
                        db.update_status(aid, "ACKNOWLEDGED", note="Auto-acknowledged for watchlist")
                    db.update_status(aid, "WATCHING", note="Pinned to watchlist")
                    save_investigation(
                        trajectory_id=tid,
                        topic_name=alert["title"],
                        lifecycle_status=status,
                        volume_path=f"{alert['post_count']} Posts",
                        coherence=alert["coherence"],
                        birth_window=alert["day"],
                        representative_post=sample_post,
                        keywords=alert["keywords"].split("; "),
                        status="Monitoring"
                    )
                    st.toast(f"Pinned #{tid} to Watchlist", icon="👁️")
                    st.rerun()

        with al_b4:
            if st.button("✕ DISMISS", key=f"btn_alt_dis_{aid}", use_container_width=True):
                db.update_status(aid, "DISMISSED", note="Operator dismissed alert")
                st.toast(f"Dismissed #{tid}", icon="✕")
                st.rerun()
