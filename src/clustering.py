import os
import re
import warnings
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Spam & Piracy Bot Regex Filter
SPAM_REGEX = re.compile(
    r'\b(crack|keygen|license key|serial key|free download|torrent|vst|pro full version|activation key|yt[0-9]|downloader pro)\b',
    re.IGNORECASE
)

# 2. Reddit Question Scaffolding & Conversational Tropes
REDDIT_TROPES = {
    # Pronouns & Auxiliaries
    "are", "was", "were", "can", "could", "would", "should", "will", "shall",
    "there", "here", "their", "they", "them", "these", "those", "this", "that",
    "who", "what", "where", "when", "why", "how", "which", "your", "you", "our",
    "my", "his", "her", "its", "him", "she", "has", "had", "have", "having",
    "been", "being", "does", "did", "doing", "done", "got", "get", "getting",
    # Reddit boilerplate
    "reddit", "askreddit", "redditor", "redditors", "post", "posts", "comment",
    "comments", "subreddit", "subreddits", "edit", "update", "deleted", "removed",
    # Conversational templates
    "thing", "things", "ever", "never", "most", "more", "least", "less", "life",
    "people", "person", "someone", "anyone", "everyone", "everybody", "nobody",
    "favorite", "favourite", "best", "worst", "good", "bad", "like", "just",
    "really", "actually", "one", "two", "three", "much", "many", "way", "day",
    "time", "make", "made", "makes", "think", "thought", "know", "knew", "want",
    "wanted", "feel", "felt", "tell", "told", "say", "said", "saying",
    "see", "saw", "seen", "look", "first", "last", "world", "year", "years",
    "something", "anything", "nothing", "everything", "all", "some", "any",
    "now", "then", "too", "also", "very", "even", "back", "out", "hey", "guys"
}

# Complete union of standard English stopwords + Reddit tropes
ALL_STOPWORDS = set(ENGLISH_STOP_WORDS).union(REDDIT_TROPES)

def extract_clean_ctf_idf(cluster_texts_map: dict, top_k: int = 4) -> dict:
    """
    Computes Class-based TF-IDF (c-TF-IDF):
    Finds keywords that are uniquely distinctive to a specific cluster
    while completely excluding boilerplate and cross-topic filler words.
    """
    cluster_vocab = {}
    all_word_doc_freq = Counter()

    for c_id, texts in cluster_texts_map.items():
        words = []
        for t in texts:
            cleaned = [
                w for w in re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
                if w not in ALL_STOPWORDS and not SPAM_REGEX.search(w)
            ]
            words.extend(cleaned)
        tf = Counter(words)
        cluster_vocab[c_id] = tf
        for unique_w in set(words):
            all_word_doc_freq[unique_w] += 1

    total_clusters = len(cluster_texts_map)
    topic_keywords = {}

    for c_id, tf in cluster_vocab.items():
        scores = {}
        total_cluster_words = sum(tf.values()) + 1
        for w, count in tf.items():
            tf_norm = count / total_cluster_words
            idf = np.log(1.0 + (total_clusters / (all_word_doc_freq[w] + 1e-9)))
            scores[w] = tf_norm * idf
        top_words = [w for w, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        topic_keywords[c_id] = top_words

    return topic_keywords

def cluster_multiweek_topics(
    embeddings_path: str = None,
    meta_path: str = None,
    output_path: str = None,
    reduced_dims: int = 12,
    min_cluster_size: int = 7
) -> pd.DataFrame:
    """
    Step 6A Part 4 (Refined):
    1. SVD Dimensionality Reduction: 384D -> 12D for sharp local density.
    2. HDBSCAN with leaf selection: Extracts tight, homogenous topics without mega-clusters.
    3. Spam Filtering: Excludes bot/software piracy posts.
    4. Quality Filtering: Prunes diffuse catch-all clusters (coherence < 0.45 or size > 150).
    5. Clean c-TF-IDF Keywords: Pure substantive topic vocabulary.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if embeddings_path is None:
        embeddings_path = os.path.join(base_dir, "data", "processed", "embeddings_6weeks.npy")
    if meta_path is None:
        meta_path = os.path.join(base_dir, "data", "processed", "embeddings_metadata_6weeks.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "processed", "clustered_topics_6weeks.csv")

    print(f"1. Loading 6-week embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(meta_path)
    print(f"   Embeddings shape: {embeddings.shape}, Total Posts: {len(df):,}")

    df["cluster_id"] = -1
    df["cluster_name"] = "Noise"

    windows = sorted(df["window_id"].unique(), key=lambda w: int(w.split("_")[1]))
    print(f"2. Running Refined Topic Discovery across {len(windows)} weekly windows...")

    cluster_summaries = []

    for win_idx, win in enumerate(windows):
        mask = (df["window_id"] == win).values
        win_emb = embeddings[mask]
        win_df = df[mask]
        week_label = win_df["week_label"].iloc[0] if "week_label" in win_df.columns else win

        if len(win_emb) < min_cluster_size:
            continue

        # Dimensionality Reduction strictly within window
        svd = TruncatedSVD(n_components=reduced_dims, random_state=42)
        reduced_emb = svd.fit_transform(win_emb)

        # HDBSCAN with leaf selection to extract tight semantic sub-topics
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,
            cluster_selection_method='leaf',
            metric='euclidean',
            copy=True
        )
        labels = clusterer.fit_predict(reduced_emb)

        unique_c = set(labels) - {-1}

        # Filter out spam clusters
        clean_clusters = {}
        for c in unique_c:
            c_posts = win_df[labels == c]["clean_text"].tolist()
            spam_ratio = sum(1 for p in c_posts if SPAM_REGEX.search(p)) / len(c_posts)
            if spam_ratio < 0.30:
                clean_clusters[c] = c_posts

        # c-TF-IDF keyword extraction
        keywords_map = extract_clean_ctf_idf(clean_clusters, top_k=4)

        accepted_window_topics = 0
        for c, c_posts in clean_clusters.items():
            c_mask = labels == c
            c_orig_emb = win_emb[c_mask]
            c_indices = win_df[c_mask].index.tolist()

            # Centroid in original 384D semantic space
            centroid = np.mean(c_orig_emb, axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

            # Measure Semantic Coherence
            coherences = cosine_similarity(c_orig_emb, centroid).flatten()
            avg_coherence = float(np.mean(coherences))

            # Quality Gate: Filter out diffuse background catch-all blobs
            if avg_coherence < 0.45 or len(c_posts) > 150:
                continue

            rep_idx = int(np.argmax(coherences))
            rep_post = c_posts[rep_idx]

            kw = keywords_map.get(c, [])
            topic_label = " / ".join(kw[:3]) if kw else f"Topic_{c}"
            full_cluster_key = f"{win}_C{c}"

            df.loc[c_indices, "cluster_id"] = c
            df.loc[c_indices, "cluster_name"] = topic_label

            cluster_summaries.append({
                "window_id": win,
                "window_index": win_idx,
                "week_label": week_label,
                "cluster_key": full_cluster_key,
                "cluster_label": topic_label,
                "topic_keywords": ", ".join(kw),
                "post_count": len(c_posts),
                "avg_semantic_coherence": round(avg_coherence, 4),
                "representative_post": rep_post
            })
            accepted_window_topics += 1

        print(f"   Window {win} ({week_label}): Discovered {accepted_window_topics} high-coherence topics.")

    # Save labeled posts and cluster summaries
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df = pd.DataFrame(cluster_summaries)
    summary_path = os.path.join(base_dir, "data", "processed", "clusters_summary_6weeks.csv")

    df.to_csv(output_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n3. Saved labeled posts to: {output_path}")
    print(f"   Saved refined topic clusters summary to: {summary_path}")

    # Summary table
    print("\n--- Summary of High-Quality Topics by Week ---")
    topic_distribution = summary_df.groupby(["window_index", "window_id", "week_label"]).agg(
        topic_count=("cluster_label", "count"),
        total_clustered_posts=("post_count", "sum"),
        avg_coherence=("avg_semantic_coherence", "mean")
    ).reset_index()
    print(topic_distribution.to_string(index=False))

    # Preview sample topics
    print("\n--- Discovered Substantive Semantic Topics (Top Sample) ---")
    preview = summary_df.sort_values(by="avg_semantic_coherence", ascending=False).head(10)
    for _, row in preview.iterrows():
        print(f"\n[{row['window_id']} - {row['week_label']}] Topic: '{row['cluster_label']}' (Size: {row['post_count']}, Coherence: {row['avg_semantic_coherence']:.3f})")
        print(f"  Rep: \"{row['representative_post']}\"")

    return summary_df

if __name__ == "__main__":
    cluster_multiweek_topics()
