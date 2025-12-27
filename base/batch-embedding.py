import psycopg2
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import torch
import time

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
DB_BATCH_SIZE = 50        # rows fetched per DB query
EMBED_BATCH_SIZE = 16     # batch size for model.encode
SLEEP_BETWEEN_BATCHES = 0  # seconds (set >0 if you want to throttle)

# -------------------------
# Load model ONCE
# -------------------------
print("Loading embedding model...")

model = SentenceTransformer(
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    trust_remote_code=True,
    device="cuda",
    model_kwargs={
        "dtype": torch.bfloat16,
    },
)

model.max_seq_length = 512

# Warm-up
_ = model.encode("warmup", normalize_embeddings=True)

print("Model ready. Starting embedding job.")

# -------------------------
# Connect to DB
# -------------------------
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

while True:
    query = f"""
    SELECT
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
        break  # no more data

    texts = [row[0] for row in rows if row[0]]

    if not texts:
        offset += DB_BATCH_SIZE
        continue

    # -------------------------
    # Embed NOW
    # -------------------------
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    batch_count = len(texts)
    total_embedded += batch_count
    offset += DB_BATCH_SIZE

    elapsed = time.perf_counter() - start_time
    docs_per_sec = total_embedded / elapsed

    print(
        f"Embedded batch: {batch_count} | "
        f"Total: {total_embedded} | "
        f"Throughput: {docs_per_sec:.2f} docs/sec"
    )

    if SLEEP_BETWEEN_BATCHES > 0:
        time.sleep(SLEEP_BETWEEN_BATCHES)

conn.close()

print(f"Embedding complete. Total documents embedded: {total_embedded}")
