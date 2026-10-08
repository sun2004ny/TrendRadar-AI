"""
src/embeddings.py - Sentence-BERT dense vector generation
Uses pathlib, Python logging, and centralized config.
"""

from pathlib import Path
import time
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(
    input_path: Optional[Path] = None,
    output_embeddings_path: Optional[Path] = None,
    output_meta_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generates Sentence-BERT embeddings:
    1. Loads cleaned text dataset.
    2. Encodes each post into a 384-dimensional unit-normalized vector.
    3. Saves embeddings to .npy array and metadata index.
    """
    if input_path is None:
        input_path = ROOT_DIR / "data" / "processed" / "reddit_cleaned_6weeks.csv"
    else:
        input_path = Path(input_path)

    if output_embeddings_path is None:
        output_embeddings_path = ROOT_DIR / "data" / "processed" / "embeddings_6weeks.npy"
    else:
        output_embeddings_path = Path(output_embeddings_path)

    if output_meta_path is None:
        output_meta_path = ROOT_DIR / "data" / "processed" / "embeddings_metadata_6weeks.csv"
    else:
        output_meta_path = Path(output_meta_path)

    logger.info("Loading cleaned data from: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Total posts to embed: %d", len(df))

    logger.info("Loading Sentence-Transformer model '%s'...", model_name)
    start_model_load = time.time()
    model = SentenceTransformer(model_name)
    logger.info("Model loaded in %.2f seconds.", time.time() - start_model_load)

    logger.info("Generating embeddings for %d posts (batch size: %d)...", len(df), batch_size)
    start_time = time.time()

    texts = df["clean_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True  # Normalizes to unit length: dot product = cosine similarity
    )
    elapsed = time.time() - start_time
    logger.info("Embeddings completed in %.2f seconds (%.1f posts/sec).", elapsed, len(texts) / max(elapsed, 1e-6))
    logger.info("Embeddings shape: %s", embeddings.shape)

    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_embeddings_path, embeddings)
    df.to_csv(output_meta_path, index=False)
    logger.info("Saved embeddings to: %s", output_embeddings_path)
    logger.info("Saved metadata index to: %s", output_meta_path)

    return embeddings, df


class IncrementalFAISSIndex:
    """
    Incremental vector index backed by FAISS (IndexFlatIP).
    Supports dynamic vector additions, cosine similarity search, and persistence,
    replacing static .npy embedding arrays for live continuous ingestion.
    """

    def __init__(
        self,
        dim: int = 384,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None
    ):
        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatIP(dim)
        except ImportError:
            self.faiss = None
            self.index = None
            logger.warning("FAISS is not installed; vector indexing fallback mode.")

        self.dim = dim
        self.id_map: list = []
        self.index_path = Path(index_path) if index_path else None
        self.meta_path = Path(meta_path) if meta_path else None

        if self.index_path and self.index_path.exists():
            self.load()

    def add(self, vectors: np.ndarray, post_ids: list) -> None:
        """Adds normalized vector embeddings and associates them with post IDs."""
        if self.index is None:
            raise RuntimeError("FAISS is not available")

        v = np.asarray(vectors, dtype=np.float32)
        if len(v.shape) == 1:
            v = v.reshape(1, -1)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(norms, 1e-12)

        self.index.add(v)
        self.id_map.extend([str(pid) for pid in post_ids])

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, list]:
        """Performs cosine similarity search against existing vectors."""
        if self.index is None or self.index.ntotal == 0:
            return np.empty((len(query_vectors), 0)), [[] for _ in range(len(query_vectors))]

        q = np.asarray(query_vectors, dtype=np.float32)
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(norms, 1e-12)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q, k)

        res_ids = []
        for row in indices:
            res_ids.append([self.id_map[idx] if 0 <= idx < len(self.id_map) else "" for idx in row])

        return distances, res_ids

    def save(self, index_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> None:
        """Persists the FAISS binary index and metadata mapping to disk."""
        import json
        ip = Path(index_path) if index_path else self.index_path
        mp = Path(meta_path) if meta_path else self.meta_path

        if ip and self.index is not None:
            ip.parent.mkdir(parents=True, exist_ok=True)
            self.faiss.write_index(self.index, str(ip))
        if mp:
            mp.parent.mkdir(parents=True, exist_ok=True)
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(self.id_map, f)

    def load(self, index_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> None:
        """Loads index and metadata mapping from disk."""
        import json
        ip = Path(index_path) if index_path else self.index_path
        mp = Path(meta_path) if meta_path else self.meta_path

        if ip and ip.exists() and self.faiss is not None:
            self.index = self.faiss.read_index(str(ip))
        if mp and mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)

    def __len__(self) -> int:
        return self.index.ntotal if self.index is not None else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_embeddings()

