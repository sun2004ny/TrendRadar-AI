import sys, os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from app.streamlit_app import load_all_data

alerts_df, trajs_df, posts_df, base_df, eval_df, clusters_df = load_all_data()

for term in ['hurricane', 'onlyfans', 'vaccine', 'pfizer', 'fda', 'policy', 'bug', 'afghanistan', 'school']:
    matches = alerts_df[
        alerts_df['topic_name'].astype(str).str.lower().str.contains(term) |
        alerts_df['representative_post'].astype(str).str.lower().str.contains(term)
    ]
    print(f"=== SEARCH: {term} ({len(matches)} matches) ===")
    for _, r in matches.head(3).iterrows():
        post_snippet = str(r.get('representative_post', ''))[:60]
        print(f"  {r['trajectory_id']} | {r['lifecycle_status']} | Vol: {r['total_volume']} | Coh: {r['avg_coherence']:.3f} | Name: {r['topic_name']} | Rep: {post_snippet}")
