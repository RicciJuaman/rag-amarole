from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    trust_remote_code=True,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
)
model.max_seq_length = 512

sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    batch_size=256,
    show_progress_bar=True,
)
print(embeddings)