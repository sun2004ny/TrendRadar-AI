import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def inspect_topic_trajectories(top_n=10):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trajs_path = os.path.join(base_dir, "data", "processed", "emerging_topics_alerts_6weeks.csv")
    matches_path = os.path.join(base_dir, "data", "processed", "trajectory_matches_6weeks.csv")
    posts_path = os.path.join(base_dir, "data", "processed", "clustered_topics_6weeks.csv")

    trajs = pd.read_csv(trajs_path)
    matches = pd.read_csv(matches_path)
    posts = pd.read_csv(posts_path)

    print("=" * 80)
    print(f"   STEP 6A PART 5.2: TOPIC QUALITY INSPECTION (TOP {top_n} CANDIDATES)")
    print("=" * 80)

    for rank, (_, t) in enumerate(trajs.head(top_n).iterrows(), 1):
        t_id = t["trajectory_id"]
        t_name = t["topic_name"]
        t_status = t["lifecycle_status"]
        t_path = t["timeline_path"]
        t_vols = t["volume_path"]
        t_score = t["emergence_score"]
        avg_coh = t.get("avg_coherence", t.get("coherence_score", 0.0))
        min_coh = t.get("min_coherence", avg_coh)
        is_cand = t.get("is_vetted_candidate", False)

        t_matches = matches[matches["trajectory_id"] == t_id]
        c_keys = t_matches["cluster_key"].tolist()

        member_posts = []
        for ck in c_keys:
            win = ck.split("_C")[0]
            cid = int(ck.split("_C")[1])
            sub = posts[(posts["window_id"] == win) & (posts["cluster_id"] == cid)]
            for text in sub["clean_text"]:
                member_posts.append((win, text))

        cand_tag = "[VETTED CANDIDATE]" if is_cand else "[FILTERED OUT]"
        print(f"\n#{rank} [{t_id}] {t_status} {cand_tag} | Topic: \"{t_name}\"")
        print(f"   Emergence Score: {t_score:.3f} | Avg Coh: {avg_coh:.3f} | Min Coh: {min_coh:.3f} | Total Posts: {len(member_posts)}")
        print(f"   Timeline: {t_path}")
        print(f"   Volumes:  {t_vols}")
        print("   Member Posts Across Checkpoints:")

        by_win = {}
        for w, p in member_posts:
            by_win.setdefault(w, []).append(p)

        for w in sorted(by_win.keys()):
            items = by_win[w]
            print(f"     * [{w}] ({len(items)} posts):")
            for sample in items[:2]:
                sample_clean = " ".join(sample.split())
                print("         - " + repr(sample_clean[:120]))
            if len(items) > 2:
                print(f"         - ... and {len(items)-2} more posts")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    inspect_topic_trajectories(6)
