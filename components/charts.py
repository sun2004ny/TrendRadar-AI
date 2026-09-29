import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_trend_velocity_chart(alerts_df: pd.DataFrame, selected_tid: str = "TRAJ_284"):
    """
    Renders the clean light-mode Plotly trend velocity chart matching Section 9.
    """
    st.markdown('<div class="tr-section-title">TREND VELOCITY & DYNAMICS</div>', unsafe_allow_html=True)

    df_chart = pd.DataFrame({
        "Day": ["Aug 11", "Aug 15", "Aug 19", "Aug 23", "Aug 27", "Aug 31", "Sep 04"],
        "Hurricane Ida (TRAJ_757)": [0, 0, 0, 2, 7, 24, 42],
        "OnlyFans Ban (TRAJ_284)": [0, 11, 24, 18, 12, 6, 2],
        "Vaccine Mandate (TRAJ_456)": [1, 3, 5, 8, 14, 21, 29],
        "Platform Bug (TRAJ_102)": [0, 1, 3, 7, 13, 8, 4]
    })

    fig = go.Figure()

    cols_meta = [
        {"col": "Hurricane Ida (TRAJ_757)", "color": "#DC2626", "fill": "rgba(220, 38, 38, 0.10)"},
        {"col": "OnlyFans Ban (TRAJ_284)", "color": "#2563EB", "fill": "rgba(37, 99, 235, 0.10)"},
        {"col": "Vaccine Mandate (TRAJ_456)", "color": "#059669", "fill": "rgba(5, 150, 105, 0.10)"},
        {"col": "Platform Bug (TRAJ_102)", "color": "#D97706", "fill": "rgba(217, 119, 6, 0.10)"},
    ]

    for item in cols_meta:
        is_selected = selected_tid in item["col"]
        fig.add_trace(go.Scatter(
            x=df_chart["Day"],
            y=df_chart[item["col"]],
            mode="lines+markers",
            name=item["col"].split(" (")[0],
            line=dict(color=item["color"], width=3 if is_selected else 2, shape="spline"),
            fill="tozeroy" if is_selected else "none",
            fillcolor=item["fill"] if is_selected else None,
            marker=dict(size=7 if is_selected else 5)
        ))

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#475569", size=11)
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        font=dict(color="#475569", family="Plus Jakarta Sans")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B"))
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B"))

    st.plotly_chart(fig, width="stretch")


def render_topic_trajectory_chart(t_row: pd.Series):
    """
    Renders trajectory progression area chart for Topic Details view.
    """
    timeline = str(t_row.get("timeline_path", "D_00")).split(" -> ")
    vols = [int(v) for v in str(t_row.get("volume_path", "1")).split(" -> ") if v.isdigit()]
    if len(vols) == 1:
        vols = [vols[0], vols[0]]
        timeline = [timeline[0], f"{timeline[0]} (Current)"]

    df_t = pd.DataFrame({"Checkpoint": timeline, "Volume": vols})
    fig = px.area(
        df_t,
        x="Checkpoint",
        y="Volume",
        markers=True,
        color_discrete_sequence=["#DC2626"]
    )
    fig.update_layout(
        height=260,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color="#475569")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B"))
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B"))
    st.plotly_chart(fig, width="stretch")
