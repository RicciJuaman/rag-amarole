"""
Configuration management for RAG system.
Centralizes all configuration to avoid inconsistencies.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for embedding model."""
    name: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length: int = 512
    dtype: str = "float16"  # Options: "float16", "bfloat16", "float32"
    device: Optional[str] = None  # None = auto-detect
    normalize_embeddings: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for PostgreSQL database."""
    user: str
    password: str
    host: str
    port: str
    dbname: str
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load database config from environment variables."""
        return cls(
            user=os.getenv("DB_USER", os.getenv("user", "")),
            password=os.getenv("DB_PASSWORD", os.getenv("password", "")),
            host=os.getenv("DB_HOST", os.getenv("host", "localhost")),
            port=os.getenv("DB_PORT", os.getenv("port", "5432")),
            dbname=os.getenv("DB_NAME", os.getenv("dbname", "")),
        )
    
    def validate(self) -> None:
        """Validate that all required fields are present."""
        if not all([self.user, self.password, self.host, self.port, self.dbname]):
            raise ValueError(
                "Missing required database configuration. "
                "Please set DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME"
            )


@dataclass
class IndexConfig:
    """Configuration for FAISS indexing."""
    db_batch_size: int = 500  # Increased from 50 for better throughput
    embed_batch_size: int = 256  # GPU batch size
    index_dir: Path = Path("indexes")
    checkpoint_interval: int = 5000  # Save checkpoint every N documents
    
    def __post_init__(self):
        """Ensure index directory exists."""
        self.index_dir.mkdir(exist_ok=True)
    
    def get_index_path(self, model_name: str) -> Path:
        """Get path for FAISS index file."""
        safe_name = model_name.replace("/", "_")
        return self.index_dir / f"{safe_name}_flat_ip.index"
    
    def get_metadata_path(self, model_name: str) -> Path:
        """Get path for document ID metadata."""
        safe_name = model_name.replace("/", "_")
        return self.index_dir / f"{safe_name}_metadata.pkl"
    
    def get_texts_path(self, model_name: str) -> Path:
        """Get path for document texts (for BM25)."""
        safe_name = model_name.replace("/", "_")
        return self.index_dir / f"{safe_name}_texts.pkl"
    
    def get_checkpoint_path(self, model_name: str) -> Path:
        """Get path for checkpoint file."""
        safe_name = model_name.replace("/", "_")
        return self.index_dir / f"{safe_name}_checkpoint.pkl"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval/search."""
    top_k: int = 10
    min_similarity: float = 0.0  # Minimum cosine similarity threshold
    use_bm25: bool = True  # Whether to use hybrid retrieval
    bm25_weight: float = 0.3  # Weight for BM25 in hybrid mode (0-1)