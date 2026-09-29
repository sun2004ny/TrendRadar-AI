import os
import streamlit as st

# Navigation labels
NAV_ITEMS = [
    "OVERVIEW", "MONITORING", "TOPICS", "ALERTS",
    "RESEARCH", "WORKSPACE", "ANALYTICS", "RESEARCH LAB"
]

NAV_ICONS = {
    "OVERVIEW": "🏠",
    "MONITORING": "📡",
    "TOPICS": "🗂️",
    "ALERTS": "🚨",
    "RESEARCH": "🔬",
    "WORKSPACE": "💼",
    "ANALYTICS": "📊",
    "RESEARCH LAB": "⚗️",
}

NAV_MAP = {k: k for k in NAV_ITEMS}

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
    """Callback fired immediately on click for 1-click instant navigation."""
    chosen = st.session_state.get("main_sidebar_nav_selector")
    if chosen and chosen in NAV_MAP:
        target = NAV_MAP[chosen]
        st.session_state["nav_section"] = target
        st.session_state["_last_synced_section"] = target


def render_sidebar(current_topic: dict = None) -> str:
    """
    Renders the dark glassmorphism sidebar with:
    - Gradient brand header (📡 TrendRadar AI)
    - Icon + pill navigation with active glow state
    - Active Focus intelligence card
    - Settings panel
    """
    # ── 1. Brand Header ────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="tr-sidebar-brand-header" style="height:40px; min-height:40px; max-height:40px; margin:0 0 20px -9px; padding:0 44px 0 0; display:flex; align-items:center;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span class="tr-brand-dish-icon">📡</span>
            <div style="line-height:1.25; min-width:0;">
                <div class="tr-brand-heading">TrendRadar AI</div>
                <div class="tr-brand-subheading">Emerging Topics Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Navigation ──────────────────────────────────────────────────────────
    current_sec = st.session_state.get("nav_section", "OVERVIEW")
    expected_nav_label = SECTION_TO_NAV.get(current_sec, "OVERVIEW")

    if "main_sidebar_nav_selector" not in st.session_state:
        st.session_state["main_sidebar_nav_selector"] = expected_nav_label
        st.session_state["_last_synced_section"] = current_sec
    elif st.session_state.get("_last_synced_section") != current_sec:
        st.session_state["main_sidebar_nav_selector"] = expected_nav_label
        st.session_state["_last_synced_section"] = current_sec

    nav_idx = NAV_ITEMS.index(expected_nav_label) if expected_nav_label in NAV_ITEMS else 0

    # Render nav with icons via custom labels
    nav_labels = [f"{NAV_ICONS.get(n, '•')}  {n}" for n in NAV_ITEMS]

    selected_nav_label = st.sidebar.radio(
        label="",
        options=NAV_ITEMS,
        format_func=lambda x: f"{NAV_ICONS.get(x, '•')}  {x}",
        index=nav_idx,
        label_visibility="collapsed",
        key="main_sidebar_nav_selector",
        on_change=on_sidebar_nav_change
    )

    # Fallback direct sync
    if selected_nav_label and selected_nav_label in NAV_MAP:
        target_section = NAV_MAP[selected_nav_label]
        if st.session_state.get("nav_section") != target_section:
            st.session_state["nav_section"] = target_section
            st.session_state["_last_synced_section"] = target_section

    # ── 3. Gradient Divider ────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="tr-gradient-divider"></div>', unsafe_allow_html=True)

    # ── 4. Active Focus Card ───────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style="font-size:0.62rem; color:#6366f1; font-weight:800; text-transform:uppercase;
                letter-spacing:0.12em; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
        <span style="width:6px; height:6px; border-radius:50%; background:#34d399;
                     box-shadow:0 0 6px #34d399; display:inline-block;"></span>
        ACTIVE FOCUS
    </div>
    """, unsafe_allow_html=True)

    tid = st.session_state.get("selected_trajectory_id", "TRAJ_284")
    t_name = "content, onlyfans, sexua"
    if current_topic is not None:
        raw_name = str(current_topic.get("topic_name", "Topic Focus"))
        t_name = raw_name.replace("[", "").replace("]", "").replace("'", "")[:28]

    st.sidebar.markdown(f"""
    <div class="tr-active-focus-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:0.72rem; color:#34d399; font-weight:800;
                         font-family:'JetBrains Mono', monospace; display:flex; align-items:center; gap:6px;">
                <span>🎯</span> {tid}
            </span>
            <span class="tr-badge tr-badge-stable" style="font-size:0.55rem; padding:2px 6px;">LIVE</span>
        </div>
        <div style="font-size:0.8rem; color:#94a3b8; font-weight:500;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            {t_name}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 5. Settings ────────────────────────────────────────────────────────────
    st.sidebar.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)

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
