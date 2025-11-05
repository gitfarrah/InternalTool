import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv, find_dotenv

# Load environment variables
try:
    load_dotenv()
except AssertionError:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "docs"
VECTOR_NAME = "content_vector"

# Configuration
CSV_PATH = "data/cleaned_pages.csv"
BATCH_SIZE = 20  # Upload in batches of 20 (smaller for network reliability)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim model

def upload_vectors():
    # Load data
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} pages")

    # Initialize embedding model (force CPU to avoid MPS memory issues)
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    print("Using CPU for embeddings to avoid memory issues")

    # Connect to Qdrant with increased timeout
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60  # 60 second timeout for network operations
    )

    # Process in smaller chunks to avoid memory issues
    EMBEDDING_CHUNK_SIZE = 100  # Process 100 documents at a time
    all_embeddings = []

    print(f"Generating embeddings in chunks of {EMBEDDING_CHUNK_SIZE}...")
    for i in tqdm(range(0, len(df), EMBEDDING_CHUNK_SIZE), desc="Embedding chunks"):
        chunk_df = df.iloc[i:i + EMBEDDING_CHUNK_SIZE]

        # Prepare texts for this chunk
        texts = []
        for idx, row in chunk_df.iterrows():
            title = row.get("title", "") or ""
            content = row.get("cleaned_content", "") or ""
            # Combine title and content for embedding
            combined_text = f"{title}\n\n{content}".strip()
            texts.append(combined_text)

        # Generate embeddings for this chunk
        chunk_embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=16,  # Smaller batch size
            convert_to_numpy=True
        )
        all_embeddings.append(chunk_embeddings)

    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Upload to Qdrant in batches
    print(f"Uploading {len(embeddings)} vectors to Qdrant in batches of {BATCH_SIZE}...")
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(df), BATCH_SIZE), total=total_batches, desc="Uploading"):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]

        points = []
        for idx, (row_idx, row) in enumerate(batch_df.iterrows()):
            point = PointStruct(
                id=row_idx,  # Use DataFrame index as unique ID
                vector={VECTOR_NAME: batch_embeddings[idx].tolist()},
                payload={
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "text": row.get("cleaned_content", "")[:10000],  # Limit text to 10k chars for payload
                }
            )
            points.append(point)

        # Upload batch with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=COLLECTION,
                    points=points
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"\n⚠️  Upload failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ Failed to upload batch after {max_retries} attempts")
                    raise
    
    print(f"✓ Successfully uploaded {len(df)} vectors to collection '{COLLECTION}'")
    
    # Verify upload
    collection_info = client.get_collection(COLLECTION)
    print(f"Collection now contains {collection_info.points_count} points")

if __name__ == "__main__":
    upload_vectors()
