import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# We use all-MiniLM-L6-v2 (384 dimensions, fast and accurate semantic vectors)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

def generate_embeddings(
    input_path: str = None,
    output_embeddings_path: str = None,
    output_meta_path: str = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64
):
    """
    Step 6A Part 3: Generates Sentence-BERT embeddings for the 6-Week dataset.
    1. Loads data/processed/reddit_cleaned_6weeks.csv (5,967 posts).
    2. Encodes each post into a 384-dimensional unit-normalized vector.
    3. Saves embeddings to data/processed/embeddings_6weeks.npy and metadata.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if input_path is None:
        input_path = os.path.join(base_dir, "data", "processed", "reddit_cleaned_6weeks.csv")
    if output_embeddings_path is None:
        output_embeddings_path = os.path.join(base_dir, "data", "processed", "embeddings_6weeks.npy")
    if output_meta_path is None:
        output_meta_path = os.path.join(base_dir, "data", "processed", "embeddings_metadata_6weeks.csv")

    print(f"1. Loading 6-week cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Total posts to embed: {len(df):,}")

    print(f"2. Loading Sentence-Transformer model '{model_name}' (from local cache)...")
    start_model_load = time.time()
    model = SentenceTransformer(model_name)
    print(f"   Model loaded in {time.time() - start_model_load:.2f} seconds.")

    print(f"3. Generating embeddings for {len(df):,} posts (batch size: {batch_size})...")
    start_time = time.time()
    
    texts = df["clean_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # Normalizes to unit length: dot product = cosine similarity
    )
    elapsed = time.time() - start_time
    print(f"   Embeddings completed in {elapsed:.2f} seconds ({len(texts)/elapsed:.1f} posts/sec).")
    print(f"   Embeddings shape: {embeddings.shape} (Rows: {embeddings.shape[0]}, Dimensions: {embeddings.shape[1]})")

    # 4. Save embeddings and corresponding metadata
    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    np.save(output_embeddings_path, embeddings)
    df.to_csv(output_meta_path, index=False)
    print(f"4. Saved embeddings array to: {output_embeddings_path}")
    print(f"   Saved metadata index to: {output_meta_path}")

    return embeddings, df

if __name__ == "__main__":
    generate_embeddings()
