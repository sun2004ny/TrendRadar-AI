import streamlit as st

def render_header(current_source: str = "Reddit"):
    """
    Renders the master top intelligence header with Tailwind dark glass theme:
    Row 1: Mode pill (HISTORICAL REPLAY), Search Console, and System Telemetry
    Row 2: Multi-source stream status ribbon (Reddit, X / Twitter, Forums)
    """
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    # ── Row 1: Executive Command Strip (Mode, Search Bar, Telemetry) ──────────
    col_h_left, col_h_mid, col_h_right = st.columns([2.0, 5.2, 4.8], vertical_alignment="center")

    with col_h_left:
        st.markdown("""
        <div class="tr-header-mode-container">
            <span class="tr-pill-live"><span class="tr-pulse-dot"></span> HISTORICAL REPLAY</span>
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

    # ── Row 2: Ingestion Feeds Status Ribbon (Full-width, zero collision) ──────
    reddit_active = " active" if current_source == "Reddit" else ""
    x_active = " active" if current_source == "X" else ""
    forums_active = " active" if current_source == "Forums" else ""

    st.markdown(f"""
    <div class="tr-source-ribbon">
        <span class="tr-source-ribbon-label">DATA STREAMS</span>
        <div class="tr-source-selector">
            <span class="tr-source-chip{reddit_active}" title="Reddit Source: Approval Pending">🔴 Reddit (approval pending)</span>
            <span class="tr-source-chip{x_active}" title="X / Twitter Stream: Not Connected">𝕏 X / Twitter (not connected)</span>
            <span class="tr-source-chip{forums_active}" title="Discussion Forums: Not Connected">💬 Forums (not connected)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

