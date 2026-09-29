import os
import streamlit as st

# Navigation labels matching the exact user specification
NAV_ITEMS = [
    "OVERVIEW",
    "MONITORING",
    "TOPICS",
    "ALERTS",
    "RESEARCH",
    "WORKSPACE",
    "ANALYTICS",
    "RESEARCH LAB"
]

NAV_MAP = {
    "OVERVIEW": "OVERVIEW",
    "MONITORING": "MONITORING",
    "TOPICS": "TOPICS",
    "ALERTS": "ALERTS",
    "RESEARCH": "RESEARCH",
    "WORKSPACE": "WORKSPACE",
    "ANALYTICS": "ANALYTICS",
    "RESEARCH LAB": "RESEARCH LAB",
}

SECTION_TO_NAV = {
    "OVERVIEW": "OVERVIEW",
    "MONITORING": "MONITORING",
    "EMERGING TOPICS": "TOPICS",
    "TOPICS": "TOPICS",
    "ALERTS": "ALERTS",
    "RESEARCH": "RESEARCH",
    "AI BRIEFING": "RESEARCH",
    "ACTION HUB": "RESEARCH",
    "WORKSPACE": "WORKSPACE",
    "ANALYTICS": "ANALYTICS",
    "RESEARCH LAB": "RESEARCH LAB",
}


def on_sidebar_nav_change():
    """Callback fired immediately on click to guarantee 1-click instant navigation."""
    chosen = st.session_state.get("main_sidebar_nav_selector")
    if chosen and chosen in NAV_MAP:
        target = NAV_MAP[chosen]
        st.session_state["nav_section"] = target
        st.session_state["_last_synced_section"] = target


def render_sidebar(current_topic: dict = None) -> str:
    """
    Renders the modern, dynamic dark enterprise navigation sidebar:
    - 📡 TrendRadar AI + Emerging Topics Intelligence header aligned with close button
    - Clean navigation items with NO extra 'NAVIGATION' text or pills
    - Dynamic active illuminated state with 1-click instant response
    - Active Focus threat intelligence card
    - Bottom pinned Settings panel
    """
    # 1. Brand Header (Aligned with radio icons below, single-line subtitle, inline with close button)
    st.sidebar.markdown("""
    <div class="tr-sidebar-brand-header">
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="tr-brand-dish-icon">📡</span>
            <div style="line-height:1.25; min-width:0;">
                <div class="tr-brand-heading">
                    TrendRadar AI
                </div>
                <div class="tr-brand-subheading">
                    Emerging Topics Intelligence
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Dynamic Interactive Navigation Pill List (1-Click Instant Response, NO 'NAVIGATION' label)
    current_sec = st.session_state.get("nav_section", "OVERVIEW")
    expected_nav_label = SECTION_TO_NAV.get(current_sec, "OVERVIEW")

    # Keep radio state in session_state strictly synchronized with current_sec
    if "main_sidebar_nav_selector" not in st.session_state:
        st.session_state["main_sidebar_nav_selector"] = expected_nav_label
        st.session_state["_last_synced_section"] = current_sec
    elif st.session_state.get("_last_synced_section") != current_sec:
        st.session_state["main_sidebar_nav_selector"] = expected_nav_label
        st.session_state["_last_synced_section"] = current_sec

    nav_idx = NAV_ITEMS.index(expected_nav_label) if expected_nav_label in NAV_ITEMS else 0

    selected_nav_label = st.sidebar.radio(
        label="",
        options=NAV_ITEMS,
        index=nav_idx,
        label_visibility="collapsed",
        key="main_sidebar_nav_selector",
        on_change=on_sidebar_nav_change
    )

    # Direct immediate update fallback
    if selected_nav_label and selected_nav_label in NAV_MAP:
        target_section = NAV_MAP[selected_nav_label]
        if st.session_state.get("nav_section") != target_section:
            st.session_state["nav_section"] = target_section
            st.session_state["_last_synced_section"] = target_section

    # 3. Active Focus Threat Intelligence Card
    st.sidebar.markdown("""
    <div style="margin-top: 18px; padding-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
        <div style="font-size:0.7rem; color:#64748B; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
            <span style="width:4px; height:4px; border-radius:50%; background:#10B981;"></span>
            ACTIVE FOCUS
        </div>
    </div>
    """, unsafe_allow_html=True)

    tid = st.session_state.get("selected_trajectory_id", "TRAJ_284")
    t_name = "content, onlyfans, sexua"
    if current_topic is not None:
        raw_name = str(current_topic.get("topic_name", "Topic Focus"))
        t_name = raw_name.replace("[", "").replace("]", "").replace("'", "")[:26]

    st.sidebar.markdown(f"""
    <div style="background:linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%); border:1px solid rgba(16, 185, 129, 0.35); border-radius:10px; padding:12px 14px; box-shadow:0 4px 16px rgba(16, 185, 129, 0.08);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.74rem; color:#10B981; font-weight:800; font-family:'JetBrains Mono', monospace; display:flex; align-items:center; gap:6px;">
                <span>🎯</span>
                {tid}
            </span>
        </div>
        <div style="font-size:0.82rem; color:#94A3B8; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:6px;">
            {t_name}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Bottom Pinned Section: Settings
    st.sidebar.markdown("""
    <div style="margin-top: 24px; padding-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar.expander("⚙️ System Settings"):
        gemini_key_in = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            help="Optional. Offline synthesis fallback active if unset."
        )
        if gemini_key_in != st.session_state.get("api_key", ""):
            st.session_state["api_key"] = gemini_key_in
        st.caption("🔒 Dense Heuristic Synthesizer active by default.")

    return st.session_state["nav_section"]

