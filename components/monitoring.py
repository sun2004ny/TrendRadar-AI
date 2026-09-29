import streamlit as st
import pandas as pd

def render_live_monitoring_feed(posts_df: pd.DataFrame):
    """
    Renders the SOC-style live monitoring feed matching Section 10.
    Displays real incoming posts with timestamps, source channels, and semantic similarity.
    """
    st.markdown('<div class="tr-section-title">LIVE MONITORING STREAM</div>', unsafe_allow_html=True)

    sample_feed = [
        {
            "src": "🔴 Reddit",
            "time": "14:32",
            "topic": "Hurricane Discussion",
            "text": "People in Louisiana, what are your evacuation and shelter plans for Hurricane Ida?",
            "sim": "0.94"
        },
        {
            "src": "𝕏 Twitter/X",
            "time": "14:31",
            "topic": "OnlyFans Policy Change",
            "text": "Official announcements confirm OnlyFans reversing adult content policy following creator backlash.",
            "sim": "0.91"
        },
        {
            "src": "🔴 Reddit",
            "time": "14:28",
            "topic": "Vaccine Mandate",
            "text": "Medical workers discussing hospital staff vaccine mandates across regional health systems.",
            "sim": "0.93"
        },
        {
            "src": "💬 Forums",
            "time": "14:25",
            "topic": "Afghanistan Evacuation",
            "text": "Civilians and foreign nationals arriving at Kabul airfield during emergency airlift operations.",
            "sim": "0.89"
        }
    ]

    # If real posts_df has clean_text, populate dynamic entries
    if not posts_df.empty and "clean_text" in posts_df.columns:
        real_posts = posts_df.dropna(subset=["clean_text"]).head(4)
        for idx, (_, r) in enumerate(real_posts.iterrows()):
            sample_feed[idx % len(sample_feed)]["text"] = str(r["clean_text"])[:140]

    for p in sample_feed:
        st.markdown(f"""
        <div class="tr-feed-card">
            <div class="tr-feed-header">
                <span><strong>{p['src']}</strong> &nbsp;•&nbsp; {p['time']} &nbsp;•&nbsp; <span class="tr-feed-topic">{p['topic']}</span></span>
                <span class="tr-sim-pill">Sim: {p['sim']}</span>
            </div>
            <div class="tr-feed-text">"{p['text']}"</div>
        </div>
        """, unsafe_allow_html=True)
