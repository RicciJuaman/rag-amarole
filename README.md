# RAG System for Amazon Reviews

A production-ready Retrieval-Augmented Generation (RAG) system for semantic search over product reviews.

## Features

- ✅ **Semantic search** using FAISS and sentence-transformers
- ✅ **Robust error handling** and logging
- ✅ **Checkpoint/resume** functionality for large datasets
- ✅ **GPU acceleration** with automatic fallback to CPU
- ✅ **Configurable** through environment variables and config file
- ✅ **Batch processing** for efficient indexing
- ✅ **Hybrid retrieval** support (semantic + BM25, optional)

## Model Specifications

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **Embedding Dimension**: 768
- **Vector Index**: FAISS (IndexFlatIP for cosine similarity)
- **Lexical Scoring**: BM25 (optional, for hybrid retrieval)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for speed)
- PostgreSQL database with reviews table

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo>
   cd rag-amarole
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   # Database configuration
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=your_database
   ```

## Usage

### 1. Build the Index

Run the indexing script to embed all documents and build the FAISS index:

```bash
python indexer.py
```

**What it does**:
- Connects to your PostgreSQL database
- Fetches documents in batches
- Generates embeddings using all-mpnet-base-v2
- Builds a FAISS index for fast similarity search
- Saves checkpoints every 5,000 documents
- Saves the final index to `indexes/`

**Resume from checkpoint**:
If the process is interrupted, simply run the command again. It will automatically resume from the last checkpoint.

**Configuration**: Edit `config.py` to adjust:
- Batch sizes
- Checkpoint frequency
- Model settings
- Index directory

### 2. Search the Index

#### Option A: CLI Search

Use the command-line search tool:

```bash
# Interactive mode
python search.py

# Single query
python search.py "What are customers saying about quality?"
```

#### Option B: Python API

Use the retriever directly in your code:

```python
from retriever import FAISSRetriever
from config import ModelConfig, IndexConfig, RetrievalConfig

# Initialize retriever
retriever = FAISSRetriever(
    model_config=ModelConfig(),
    index_config=IndexConfig(),
    retrieval_config=RetrievalConfig(top_k=10)
)

# Search
results = retriever.search("What are customers saying about quality?")

# Print results
for result in results:
    print(f"Doc ID: {result.doc_id}, Score: {result.score:.4f}")
```

#### Option C: REST API (NEW!)

Start the FastAPI server:

```bash
# Development mode
uvicorn api:app --reload

# Or with make
make api-dev
```

Then search using HTTP:

```bash
# Using curl
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "product quality", "top_k": 5}'

# Using Python client
python api_client.py
```

**API Features**:
- ✅ **Full metadata** - Returns Summary and Text for each result
- ✅ **RESTful** - Standard HTTP endpoints
- ✅ **Interactive docs** - Auto-generated at `/docs`
- ✅ **Fast** - Async request handling
- ✅ **Production-ready** - CORS, error handling, logging

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

### 3. Batch Search

Search multiple queries at once for better efficiency:

```python
queries = [
    "product quality issues",
    "shipping and delivery",
    "customer service experience"
]

all_results = retriever.batch_search(queries, top_k=5)

for query, results in zip(queries, all_results):
    print(f"\nQuery: {query}")
    for result in results:
        print(f"  - Doc {result.doc_id}: {result.score:.4f}")
```

## Project Structure

```
rag-amarole/
├── config.py              # Configuration management
├── indexer.py             # Embedding and FAISS indexing
├── retriever.py           # Search and retrieval
├── api.py                 # FastAPI REST API (NEW!)
├── api_client.py          # API client example (NEW!)
├── search.py              # CLI search tool
├── validate.py            # System validation
├── example.py             # Usage examples
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── indexes/               # Generated FAISS indexes and metadata
│   ├── *.index           # FAISS index files
│   ├── *_metadata.pkl    # Document ID mappings
│   └── *_checkpoint.pkl  # Resume checkpoints
└── base/                  # Legacy code (for reference)
```

## Configuration Options

### ModelConfig
```python
model_name: str = "sentence-transformers/all-mpnet-base-v2"
max_seq_length: int = 512
dtype: str = "float16"  # "float16" | "bfloat16" | "float32"
device: Optional[str] = None  # None = auto-detect
normalize_embeddings: bool = True
```

### IndexConfig
```python
db_batch_size: int = 500           # Documents per DB query
embed_batch_size: int = 256        # GPU batch size
checkpoint_interval: int = 5000    # Checkpoint frequency
index_dir: Path = Path("indexes")  # Output directory
```

### RetrievalConfig
```python
top_k: int = 10                    # Number of results
min_similarity: float = 0.0        # Similarity threshold
use_bm25: bool = False             # Enable hybrid retrieval
bm25_weight: float = 0.3           # BM25 weight (0-1)
```

## Performance Tips

1. **GPU Usage**: The system automatically uses GPU if available. Monitor with:
   ```bash
   nvidia-smi
   ```

2. **Batch Sizes**: 
   - Increase `db_batch_size` (500-1000) for fewer DB round trips
   - Increase `embed_batch_size` (256-512) if you have more GPU memory

3. **Checkpointing**: 
   - Lower `checkpoint_interval` if your system is unstable
   - Higher values = less overhead but more loss if interrupted

4. **Memory**: For large datasets (millions of docs), consider:
   - Using `faiss-gpu` for GPU-accelerated search
   - Using IVF indexes instead of Flat for faster search
   - Splitting into multiple indexes

## Database Schema

Expected schema for the `reviews` table:

```sql
CREATE TABLE reviews (
    "Id" INTEGER PRIMARY KEY,
    "Summary" TEXT,
    "Text" TEXT
);
```

## Troubleshooting

### "Database connection failed"
- Check your `.env` file has correct credentials
- Ensure PostgreSQL is running
- Verify network connectivity

### "Index not found"
- Run `python indexer.py` first to build the index

### "CUDA out of memory"
- Reduce `embed_batch_size` in `config.py`
- Use `dtype="float16"` instead of `float32`
- Use CPU mode by setting `device="cpu"`

### Slow indexing
- Check GPU is being used: look for "Auto-detected device: cuda"
- Increase batch sizes
- Ensure database is indexed on `Id` column

## Future Enhancements

- [ ] Add support for filtering by metadata
- [ ] Implement reranking with cross-encoder
- [ ] Add API endpoint for production deployment
- [ ] Support for incremental index updates
- [ ] Add evaluation metrics and benchmarks

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.