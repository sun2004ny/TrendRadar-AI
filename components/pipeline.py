import streamlit as st

def render_pipeline_loop():
    """
    Renders the bottom technical pipeline process loop matching the reference design.
    """
    st.markdown("""
    <div class="tr-pipeline-loop">
        <div class="tr-pipe-node">NEW POSTS</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">INGEST</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">CLEAN</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">SEMANTIC EMBEDDING</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">CLUSTER</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">IDENTIFY TOPIC</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">DRIFT GATE</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">GROWTH CALCULATION</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node tr-pipe-decision">DECISION (EMERGING / INCUBATION / FILTERED)</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">UPDATE TOPIC</div>
        <div class="tr-pipe-arrow">→</div>
        <div class="tr-pipe-node">MONITOR AGAIN</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("TrendRadar AI • Causal Topic Emergence Detection (research prototype) • Powered by Dense Semantic NLP")
