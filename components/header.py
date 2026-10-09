import streamlit as st

def render_header(current_source: str = "Reddit"):
    """
    Renders the master top intelligence header with Tailwind dark glass theme:
    - HISTORICAL REPLAY animated pill with pulse dot
    - Multi-source selector chips (Reddit, X, Forums)
    - Search input field
    - Operational status telemetry (Last updated, Alerts, User)
    """
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    col_h_left, col_h_mid, col_h_right = st.columns([5.5, 3.2, 3.3], vertical_alignment="center")

    reddit_active = " active" if current_source == "Reddit" else ""
    x_active = " active" if current_source == "X" else ""
    forums_active = " active" if current_source == "Forums" else ""

    with col_h_left:
        st.markdown(f"""
        <div class="tr-brand-group">
            <span class="tr-pill-live"><span class="tr-pulse-dot"></span> HISTORICAL REPLAY</span>
            <div class="tr-source-selector">
                <span class="tr-source-chip{reddit_active}">🔴 Reddit (approval pending)</span>
                <span class="tr-source-chip{x_active}">𝕏 X / Twitter (not connected)</span>
                <span class="tr-source-chip{forums_active}">💬 Forums (not connected)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_h_mid:
        search_val = st.text_input(
            "Search Console",
            value=st.session_state["search_query"],
            placeholder="🔍 Search topics, keywords, entities...",
            label_visibility="collapsed",
            key="header_search_input"
        )
        if search_val != st.session_state["search_query"]:
            st.session_state["search_query"] = search_val

    acknowledged = st.session_state.get("acknowledged_alerts", set())
    dismissed = st.session_state.get("dismissed_alerts", set())
    critical_tids = {"TRAJ_284", "TRAJ_757", "TRAJ_456"}
    rem_alerts = max(0, len(critical_tids - acknowledged - dismissed))

    with col_h_right:
        st.markdown(f"""
        <div class="tr-meta-status">
            <span class="tr-meta-chip">🕐 <strong>10s ago</strong></span>
            <span class="tr-meta-alert">🔔 {rem_alerts} Alerts</span>
            <span class="tr-meta-user">👤 Analyst Admin</span>
        </div>
        """, unsafe_allow_html=True)
