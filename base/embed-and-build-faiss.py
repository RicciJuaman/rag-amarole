import os
import time
import pickle
import numpy as np
import psycopg2
import faiss
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# ENV + CONFIG
# -------------------------------------------------
load_dotenv()

DB_BATCH_SIZE = 50          # rows fetched from DB per query
EMBED_BATCH_SIZE = 250      # GPU batch size
EMBED_DIM = 896             # KaLM v2.5 dimension

INDEX_DIR = "indexes"
INDEX_PATH = os.path.join(INDEX_DIR, "kalm_flat_ip.index")
META_PATH = os.path.join(INDEX_DIR, "doc_ids.pkl")

os.makedirs(INDEX_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD MODEL (GPU)
# -------------------------------------------------
print("Loading KaLM embedding model on GPU...")

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    trust_remote_code=True,
    device="cuda",
    model_kwargs={"dtype": torch.float16},
)

model.max_seq_length = 512

# Warm-up
_ = model.encode("warmup", normalize_embeddings=True)

print("Model ready.")

# -------------------------------------------------
# FAISS INDEX (BASELINE)
# -------------------------------------------------
index = faiss.IndexFlatIP(EMBED_DIM)
doc_ids = []

# -------------------------------------------------
# DB CONNECTION
# -------------------------------------------------
conn = psycopg2.connect(
    user=os.getenv("user"),
    password=os.getenv("password"),
    host=os.getenv("host"),
    port=os.getenv("port"),
    dbname=os.getenv("dbname"),
)

offset = 0
total_embedded = 0
start_time = time.perf_counter()

# -------------------------------------------------
# INGEST + EMBED LOOP
# -------------------------------------------------
print("Starting embedding + indexing job...")

while True:
    query = f"""
    SELECT
      "Id",
      TRIM(
        COALESCE("Summary", '') || E'\\n\\n' || COALESCE("Text", '')
      ) AS combined_text
    FROM reviews
    LIMIT {DB_BATCH_SIZE}
    OFFSET {offset};
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        break

    batch_ids = []
    batch_texts = []

    for r in rows:
        if r[1]:
            batch_ids.append(r[0])
            batch_texts.append(r[1])

    if not batch_texts:
        offset += DB_BATCH_SIZE
        continue

    # ---- EMBED ON GPU ----
    embeddings = model.encode(
        batch_texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index.add(embeddings)
    doc_ids.extend(batch_ids)

    batch_count = len(batch_texts)
    total_embedded += batch_count
    offset += DB_BATCH_SIZE

    elapsed = time.perf_counter() - start_time
    docs_per_sec = total_embedded / elapsed

    print(
        f"Embedded batch: {batch_count} | "
        f"Total: {total_embedded} | "
        f"Throughput: {docs_per_sec:.2f} docs/sec"
    )

conn.close()

# -------------------------------------------------
# SAVE INDEX + METADATA
# -------------------------------------------------
faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(doc_ids, f)

print("\n✅ Embedding job complete.")
print(f"Total vectors indexed: {index.ntotal}")
print(f"FAISS index saved to: {INDEX_PATH}")
print(f"Document ID map saved to: {META_PATH}")
