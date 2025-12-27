import psycopg2
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import torch
import time

load_dotenv()

# -------------------------
# Measure DB query latency
# -------------------------
t0 = time.perf_counter()

conn = psycopg2.connect(
    user=os.getenv("user"),
    password=os.getenv("password"),
    host=os.getenv("host"),
    port=os.getenv("port"),
    dbname=os.getenv("dbname"),
)

query = """
SELECT
  TRIM(
    COALESCE("Summary", '') || E'\\n\\n' || COALESCE("Text", '')
  ) AS combined_text
FROM reviews
LIMIT 1;
"""

with conn.cursor() as cur:
    cur.execute(query)
    row = cur.fetchone()

conn.close()

t1 = time.perf_counter()
db_latency_ms = (t1 - t0) * 1000

if row is None:
    raise ValueError("No rows returned from database")

combined_text = row[0]

print("=== TEXT TO EMBED ===")
print(combined_text)
print("=====================")

print(f"DB query latency: {db_latency_ms:.2f} ms")

# -------------------------
# Load embedding model
# -------------------------
model = SentenceTransformer(
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    trust_remote_code=True,
    model_kwargs={
        "dtype": torch.bfloat16,
    },
)

model.max_seq_length = 512
print("Model loaded.")

# -------------------------
# Warm-up (important)
# -------------------------
_ = model.encode("warmup", normalize_embeddings=True)

# -------------------------
# Measure embedding latency
# -------------------------
t2 = time.perf_counter()

embedding = model.encode(
    combined_text,
    normalize_embeddings=True,
)

t3 = time.perf_counter()
embedding_latency_ms = (t3 - t2) * 1000
total_latency_ms = (t3 - t0) * 1000

print("Embedding shape:", embedding.shape)
print("Vector norm:", float((embedding @ embedding) ** 0.5))

print(f"Embedding latency: {embedding_latency_ms:.2f} ms")
print(f"Total end-to-end latency: {total_latency_ms:.2f} ms")
