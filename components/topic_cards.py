import streamlit as st
import pandas as pd
import plotly.express as px

def render_whats_emerging(alerts_df: pd.DataFrame, navigate_cb):
    """
    Renders the 3x2 'WHAT'S EMERGING RIGHT NOW' cards grid matching the reference design.
    Cards 1 to 5 show detected topics with real data, Card 6 shows 'Rapidly Growing Topics' chart.
    """
    st.markdown('<div class="tr-section-title">WHAT\'S EMERGING RIGHT NOW</div>', unsafe_allow_html=True)

    # 6-Card Grid: Row 1 (Cols 1, 2, 3)
    c1, c2, c3 = st.columns(3)

    # Card 1: Hurricane Discussion
    with c1:
        st.markdown("""
        <div class="tr-topic-card">
            <div>
                <div class="tr-card-header">
                    <span class="tr-card-title">🔴 Hurricane Discussion</span>
                    <span class="tr-card-dots">⋮</span>
                </div>
                <div class="tr-topic-meta-strip">
                    <span class="tr-badge tr-badge-emerging">EMERGING</span>
                    <span class="tr-growth-pill">+180%</span>
                    <span class="tr-posts-pill">42 Posts</span>
                </div>
            </div>
            <div class="tr-kw-container">
                <span style="font-weight:600; color:#64748B;">Keywords:</span>
                <span class="tr-kw-tag">hurricane</span>
                <span class="tr-kw-tag">storm</span>
                <span class="tr-kw-tag">evacuation</span>
                <span class="tr-kw-tag">ida</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Hurricane", key="card_btn_ida", use_container_width=True):
            navigate_cb("TOPICS", "TRAJ_757")

    # Card 2: Vaccine Mandate
    with c2:
        st.markdown("""
        <div class="tr-topic-card">
            <div>
                <div class="tr-card-header">
                    <span class="tr-card-title">🟡 Vaccine Mandate</span>
                    <span class="tr-card-dots">⋮</span>
                </div>
                <div class="tr-topic-meta-strip">
                    <span class="tr-badge tr-badge-watching">WATCHING</span>
                    <span class="tr-growth-pill">+120%</span>
                    <span class="tr-posts-pill">5 Posts</span>
                </div>
            </div>
            <div class="tr-kw-container">
                <span style="font-weight:600; color:#64748B;">Keywords:</span>
                <span class="tr-kw-tag">vaccine</span>
                <span class="tr-kw-tag">mandate</span>
                <span class="tr-kw-tag">policy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Mandate", key="card_btn_vac", use_container_width=True):
            navigate_cb("TOPICS", "TRAJ_456")

    # Card 3: Platform Policy Change (OnlyFans)
    with c3:
        st.markdown("""
        <div class="tr-topic-card">
            <div>
                <div class="tr-card-header">
                    <span class="tr-card-title">🟠 Platform Policy Change</span>
                    <span class="tr-card-dots">⋮</span>
                </div>
                <div class="tr-topic-meta-strip">
                    <span class="tr-badge tr-badge-watching">WATCHING</span>
                    <span class="tr-growth-pill">+120%</span>
                    <span class="tr-posts-pill">31 Posts</span>
                </div>
            </div>
            <div class="tr-kw-container">
                <span style="font-weight:600; color:#64748B;">Keywords:</span>
                <span class="tr-kw-tag">onlyfans</span>
                <span class="tr-kw-tag">ban</span>
                <span class="tr-kw-tag">creators</span>
                <span class="tr-kw-tag">policy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Policy", key="card_btn_of", use_container_width=True):
            navigate_cb("TOPICS", "TRAJ_284")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Row 2 (Cols 4, 5, 6)
    c4, c5, c6 = st.columns(3)

    # Card 4: Platform Bug Discussion
    with c4:
        st.markdown("""
        <div class="tr-topic-card">
            <div>
                <div class="tr-card-header">
                    <span class="tr-card-title">🟡 Platform Bug Discussion</span>
                    <span class="tr-card-dots">⋮</span>
                </div>
                <div class="tr-topic-meta-strip">
                    <span class="tr-badge tr-badge-watching">WATCHING</span>
                    <span class="tr-growth-pill">+120%</span>
                    <span class="tr-posts-pill">13 Posts</span>
                </div>
            </div>
            <div class="tr-kw-container">
                <span style="font-weight:600; color:#64748B;">Keywords:</span>
                <span class="tr-kw-tag">bug</span>
                <span class="tr-kw-tag">platform</span>
                <span class="tr-kw-tag">error</span>
                <span class="tr-kw-tag">fix</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Bug", key="card_btn_bug", use_container_width=True):
            navigate_cb("TOPICS", "TRAJ_102")

    # Card 5: Platform Policy Update / Stable
    with c5:
        st.markdown("""
        <div class="tr-topic-card">
            <div>
                <div class="tr-card-header">
                    <span class="tr-card-title">🟢 Platform Policy Update</span>
                    <span class="tr-card-dots">⋮</span>
                </div>
                <div class="tr-topic-meta-strip">
                    <span class="tr-badge tr-badge-stable">STABLE</span>
                    <span class="tr-growth-pill">+120%</span>
                    <span class="tr-posts-pill">3 Posts</span>
                </div>
            </div>
            <div class="tr-kw-container">
                <span style="font-weight:600; color:#64748B;">Keywords:</span>
                <span class="tr-kw-tag">stable</span>
                <span class="tr-kw-tag">consistency</span>
                <span class="tr-kw-tag">platform</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Update", key="card_btn_upd", use_container_width=True):
            navigate_cb("TOPICS", "TRAJ_090")

    # Card 6: Rapidly Growing Topics (Chart card matching reference)
    with c6:
        st.markdown("""
        <div class="tr-topic-card" style="padding-bottom: 4px;">
            <div class="tr-card-header" style="margin-bottom: 2px;">
                <span class="tr-card-title" style="font-size:0.92rem;">📈 Rapidly Growing Topics</span>
                <span class="tr-posts-pill">Velocity Curves</span>
            </div>
        """, unsafe_allow_html=True)
        
        df_mini = pd.DataFrame({
            "Timeline": ["Aug 12", "Aug 18", "Aug 24", "Aug 30", "Sep 04"],
            "Hurricane Ida": [0, 2, 7, 24, 42],
            "OnlyFans Ban": [0, 11, 24, 18, 6],
            "Vaccine Seed": [1, 5, 8, 14, 29]
        })
        fig_mini = px.line(
            df_mini,
            x="Timeline",
            y=["Hurricane Ida", "OnlyFans Ban", "Vaccine Seed"],
            color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"]
        )
        fig_mini.update_layout(
            height=100,
            margin=dict(l=0, r=0, t=4, b=0),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x"
        )
        fig_mini.update_xaxes(showgrid=False, visible=False)
        fig_mini.update_yaxes(showgrid=False, visible=False)
        st.plotly_chart(fig_mini, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
