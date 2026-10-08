"""
src/clustering.py - HDBSCAN density-based topic clustering and c-TF-IDF keyword extraction
Uses pathlib, Python logging, and centralized config.
"""

from pathlib import Path
import re
import warnings
import logging
from collections import Counter
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import ROOT_DIR, get_config

logger = logging.getLogger(__name__)
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


def extract_clean_ctf_idf(cluster_texts_map: Dict[Any, List[str]], top_k: Optional[int] = None) -> Dict[Any, List[str]]:
    """
    Computes Class-based TF-IDF (c-TF-IDF):
    Finds keywords uniquely distinctive to a specific cluster
    while excluding boilerplate and cross-topic filler words.
    """
    cfg = get_config()
    k = top_k if top_k is not None else int(cfg.clustering.get("c_tf_idf_top_k", 4))

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
        top_words = [w for w, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]
        topic_keywords[c_id] = top_words

    return topic_keywords


def cluster_multiweek_topics(
    embeddings_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    reduced_dims: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Density-based clustering with HDBSCAN and c-TF-IDF:
    1. SVD Dimensionality Reduction: 384D -> 12D for sharp local density.
    2. HDBSCAN with leaf selection.
    3. Quality Filtering: Prunes diffuse catch-all clusters.
    4. Clean c-TF-IDF Keywords.
    """
    cfg = get_config()
    dims = reduced_dims or int(cfg.clustering.get("svd_components", 12))
    min_c_size = min_cluster_size or int(cfg.clustering.get("min_cluster_size", 7))
    min_samp = min_samples or int(cfg.clustering.get("min_samples", 3))
    spam_thresh = float(cfg.clustering.get("spam_ratio_threshold", 0.35))
    max_c_size = int(cfg.clustering.get("max_cluster_size", 150))
    coh_min = float(cfg.clustering.get("coherence_min", 0.45))
    top_k = int(cfg.clustering.get("c_tf_idf_top_k", 4))

    if embeddings_path is None:
        embeddings_path = ROOT_DIR / "data" / "processed" / "embeddings_6weeks.npy"
    else:
        embeddings_path = Path(embeddings_path)

    if meta_path is None:
        meta_path = ROOT_DIR / "data" / "processed" / "embeddings_metadata_6weeks.csv"
    else:
        meta_path = Path(meta_path)

    if output_path is None:
        output_path = ROOT_DIR / "data" / "processed" / "clustered_topics_6weeks.csv"
    else:
        output_path = Path(output_path)

    logger.info("Loading embeddings from: %s", embeddings_path)
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(meta_path)
    logger.info("Embeddings shape: %s, Total Posts: %d", embeddings.shape, len(df))

    df["cluster_id"] = -1
    df["cluster_name"] = "Noise"

    windows = sorted(df["window_id"].unique(), key=lambda w: int(w.split("_")[1]))
    logger.info("Running Topic Discovery across %d windows...", len(windows))

    for win_idx, win in enumerate(windows):
        mask = (df["window_id"] == win).values
        win_emb = embeddings[mask]
        win_df = df[mask]

        if len(win_emb) < min_c_size:
            continue

        # SVD Dimensionality Reduction strictly within window
        svd = TruncatedSVD(n_components=dims, random_state=42)
        reduced_emb = svd.fit_transform(win_emb)

        # HDBSCAN with leaf selection
        clusterer = HDBSCAN(
            min_cluster_size=min_c_size,
            min_samples=min_samp,
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
            if spam_ratio < spam_thresh and len(c_posts) <= max_c_size:
                clean_clusters[c] = c_posts

        # c-TF-IDF keyword extraction
        keywords_map = extract_clean_ctf_idf(clean_clusters, top_k=top_k)

        for c, c_posts in clean_clusters.items():
            c_mask = labels == c
            c_indices = np.where(c_mask)[0]
            c_vectors = win_emb[c_indices]
            centroid = np.mean(c_vectors, axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            sims = cosine_similarity(c_vectors, centroid).flatten()
            avg_coherence = float(np.mean(sims))

            if avg_coherence < coh_min:
                labels[c_mask] = -1
                continue

            kws = keywords_map.get(c, [f"topic_{c}"])
            df.loc[df[mask].index[c_mask], "cluster_id"] = c
            df.loc[df[mask].index[c_mask], "cluster_name"] = " / ".join(kws[:3])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved clustered topics to: %s", output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cluster_multiweek_topics()
