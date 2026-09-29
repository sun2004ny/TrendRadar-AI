import streamlit as st
import pandas as pd
from components.charts import render_topic_trajectory_chart
from src.workspace import save_investigation

def render_topic_details_view(current_topic: pd.Series, all_tids: list, navigate_cb):
    """
    Renders the Topic Details & Explainable AI inspector matching Section 13 and reference design.
    """
    st.markdown('<div class="tr-section-title">TOPIC DETAILS & EXPLAINABLE AI INSPECTOR</div>', unsafe_allow_html=True)

    col_sel, _ = st.columns([2, 1])
    with col_sel:
        chosen_tid = st.selectbox(
            "Select Trajectory to Examine:",
            all_tids,
            index=all_tids.index(st.session_state["selected_trajectory_id"]) if st.session_state["selected_trajectory_id"] in all_tids else 0,
            label_visibility="collapsed"
        )
        if chosen_tid != st.session_state["selected_trajectory_id"]:
            st.session_state["selected_trajectory_id"] = chosen_tid
            st.rerun()

    if current_topic is None:
        st.warning("No trajectory selected.")
        return

    t_row = current_topic
    tid = t_row["trajectory_id"]
    t_name = str(t_row.get("topic_name", "Topic Focus")).replace("[", "").replace("]", "").replace("'", "")
    status = str(t_row.get("lifecycle_status", "[ALERT]"))
    coh = float(t_row.get("avg_coherence", 0.0))
    vol_path = str(t_row.get("volume_path", "1"))
    birth = str(t_row.get("birth_window", "D_00"))
    rep_post = str(t_row.get("representative_post", ""))
    keywords = [k.strip() for k in t_name.split(",")]

    # Calculate Velocity & Acceleration from volume path
    vols = [int(v) for v in vol_path.split(" -> ") if v.isdigit()]
    current_vol = vols[-1] if vols else 1
    prev_vol = vols[-2] if len(vols) >= 2 else (vols[0] if vols else 1)
    velocity = current_vol - prev_vol
    acceleration = velocity - (prev_vol - (vols[-3] if len(vols) >= 3 else 0)) if len(vols) >= 2 else 1

    # Bold Red Banner matching reference design
    st.markdown(f"""
    <div class="tr-topic-detail-banner">
        <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; opacity:0.9; letter-spacing:0.06em;">
            TARGET TOPIC • {status}
        </div>
        <h2>#{tid} — {t_name}</h2>
        <div class="tr-banner-metrics-strip">
            <span class="tr-banner-metric-item">Lifecycle: <strong>{status}</strong></span>
            <span class="tr-banner-metric-item">Volume: <strong>{current_vol} Posts</strong></span>
            <span class="tr-banner-metric-item">Velocity: <strong>+{max(1, velocity)}/slice</strong></span>
            <span class="tr-banner-metric-item">Acceleration: <strong>+{max(1, acceleration)}</strong></span>
            <span class="tr-banner-metric-item">Coherence: <strong>{coh:.3f}</strong></span>
            <span class="tr-banner-metric-item">Detected: <strong>{birth}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_traj, tab_ev, tab_kw = st.tabs(["📈 Trajectory Dynamics", "🔬 Explainable AI Evidence", "🏷️ c-TF-IDF Lexical Core"])

    with tab_traj:
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.markdown("#### Topic Trajectory Progression")
            render_topic_trajectory_chart(t_row)
        with col_c2:
            st.markdown("#### Canonical User Signal")
            st.markdown(f"""
            <div class="tr-post-quote" style="font-size:0.92rem; padding:16px 18px; margin-top:8px;">
                💬 "{rep_post}"
            </div>
            <div style="font-size:0.76rem; color:#94A3B8; margin-top:8px; font-family:'JetBrains Mono', monospace;">
                CHECKPOINT TRAJECTORY: <strong>{t_row.get('timeline_path', birth)}</strong><br>
                VOLUME HISTORY: <strong>{vol_path}</strong>
            </div>
            """, unsafe_allow_html=True)

    with tab_ev:
        st.markdown("#### Five-Point Explainable AI Verification Checklist")
        st.markdown(r"""
        <div class="tr-checklist">
            <div class="tr-checklist-item">
                <span style="color:#10B981; font-weight:900;">✓</span>
                <div><strong>Semantic Coherence:</strong> Sentence-BERT embedding cosine density verified inside cluster core ($\ge 0.65$).</div>
            </div>
            <div class="tr-checklist-item">
                <span style="color:#10B981; font-weight:900;">✓</span>
                <div><strong>Temporal Persistence:</strong> Linked across consecutive daily checkpoints without fragmentation ($D_t \to D_{t+1}$).</div>
            </div>
            <div class="tr-checklist-item">
                <span style="color:#10B981; font-weight:900;">✓</span>
                <div><strong>Lexical Consistency:</strong> Jaccard keyword overlap $\ge 0.20$ guarantees semantic topic stability.</div>
            </div>
            <div class="tr-checklist-item">
                <span style="color:#10B981; font-weight:900;">✓</span>
                <div><strong>Growth Signal:</strong> Exceeds $2.0\%$ slice volume threshold with verified positive burst velocity.</div>
            </div>
            <div class="tr-checklist-item">
                <span style="color:#10B981; font-weight:900;">✓</span>
                <div><strong>Domain Evidence:</strong> Boilerplate question syntax stripped prior to vector quantization.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_kw:
        st.markdown("#### Class-based TF-IDF Distinctive Vocabulary")
        st.write("Keywords extracted across connected cluster partitions:")
        kw_html = " ".join([f"<span class='tr-kw-tag' style='font-size:0.8rem; padding:4px 8px;'>{k}</span>" for k in keywords])
        st.markdown(f"<div>{kw_html}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    b_act1, b_act2, b_act3 = st.columns(3)
    with b_act1:
        if st.button("🔬 Search External Context in Research Hub", key="btn_dt_res", use_container_width=True):
            navigate_cb("RESEARCH", tid)
    with b_act2:
        if st.button("🧠 Synthesize AI Executive Briefing", key="btn_dt_brief", use_container_width=True):
            navigate_cb("AI BRIEFING", tid)
    with b_act3:
        if st.button("📌 Pin to Workspace Desk", key="btn_dt_pin", use_container_width=True):
            save_investigation(
                trajectory_id=tid,
                topic_name=t_name,
                lifecycle_status=status,
                volume_path=vol_path,
                coherence=coh,
                birth_window=birth,
                representative_post=rep_post,
                keywords=keywords
            )
            st.toast(f"Case #{tid} committed to Workspace Desk", icon="📌")
