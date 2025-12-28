"""
Improved embedding and FAISS indexing with error handling,
checkpointing, and proper configuration management.
"""

import logging
import pickle
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import psycopg2 # pyright: ignore[reportMissingModuleSource]
import faiss # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]

from config import ModelConfig, DatabaseConfig, IndexConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """Track indexing statistics."""
    total_documents: int = 0
    total_batches: int = 0
    start_time: float = 0
    errors: int = 0
    
    def get_throughput(self) -> float:
        """Calculate documents per second."""
        elapsed = time.perf_counter() - self.start_time
        return self.total_documents / elapsed if elapsed > 0 else 0


class EmbeddingModel:
    """Wrapper for sentence transformer model with proper setup."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self) -> SentenceTransformer:
        """Load the embedding model with proper configuration."""
        logger.info(f"Loading model: {self.config.name}")
        
        # Auto-detect device if not specified
        device = self.config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")
        
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)
        
        try:
            model = SentenceTransformer(
                self.config.name,
                trust_remote_code=True,
                device=device,
                model_kwargs={"dtype": torch_dtype},
            )
            model.max_seq_length = self.config.max_seq_length
            
            # Warm-up
            logger.info("Warming up model...")
            _ = model.encode("warmup", normalize_embeddings=True)
            
            logger.info("Model ready")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return np.asarray(embeddings, dtype="float32")
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise


class DatabaseReader:
    """Handle database connections and queries."""
    
    def __init__(self, config: DatabaseConfig):
        config.validate()
        self.config = config
        self.conn = None
    
    def connect(self) -> None:
        """Establish database connection."""
        try:
            logger.info(f"Connecting to database: {self.config.host}:{self.config.port}/{self.config.dbname}")
            self.conn = psycopg2.connect(
                user=self.config.user,
                password=self.config.password,
                host=self.config.host,
                port=self.config.port,
                dbname=self.config.dbname,
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def fetch_batch(
        self, 
        batch_size: int, 
        offset: int
    ) -> List[Tuple[int, str]]:
        """Fetch a batch of documents from the database."""
        if self.conn is None:
            raise RuntimeError("Database not connected")
        
        query = """
            SELECT
                "Id",
                TRIM(
                    COALESCE("Summary", '') || E'\\n\\n' || COALESCE("Text", '')
                ) AS combined_text
            FROM reviews
            WHERE TRIM(COALESCE("Summary", '') || E'\\n\\n' || COALESCE("Text", '')) != ''
            ORDER BY "Id"
            LIMIT %s
            OFFSET %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (batch_size, offset))
                rows = cur.fetchall()
            return [(row[0], row[1]) for row in rows if row[1]]
        except Exception as e:
            logger.error(f"Failed to fetch batch at offset {offset}: {e}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class FAISSIndexer:
    """Manage FAISS index creation and persistence."""
    
    def __init__(self, embedding_dim: int, index_config: IndexConfig, model_name: str):
        self.embedding_dim = embedding_dim
        self.config = index_config
        self.model_name = model_name
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.doc_ids = []
        
        logger.info(f"Initialized FAISS IndexFlatIP with dimension {embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, doc_ids: List[int]) -> None:
        """Add embeddings to the index."""
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("Number of embeddings must match number of doc IDs")
        
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        index_path = self.config.get_index_path(self.model_name)
        meta_path = self.config.get_metadata_path(self.model_name)
        
        try:
            logger.info(f"Saving FAISS index to {index_path}")
            faiss.write_index(self.index, str(index_path))
            
            logger.info(f"Saving metadata to {meta_path}")
            with open(meta_path, "wb") as f:
                pickle.dump(self.doc_ids, f)
            
            logger.info("Save complete")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def save_checkpoint(self, offset: int) -> None:
        """Save a checkpoint for resume capability."""
        checkpoint_path = self.config.get_checkpoint_path(self.model_name)
        checkpoint_data = {
            "offset": offset,
            "doc_ids": self.doc_ids,
            "total_vectors": self.index.ntotal,
        }
        
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved at offset {offset}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[int]:
        """Load checkpoint if exists and return last offset."""
        checkpoint_path = self.config.get_checkpoint_path(self.model_name)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            
            self.doc_ids = checkpoint["doc_ids"]
            logger.info(f"Loaded checkpoint: {checkpoint['total_vectors']} vectors at offset {checkpoint['offset']}")
            return checkpoint["offset"]
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None


def build_index(
    model_config: ModelConfig = ModelConfig(),
    db_config: DatabaseConfig = DatabaseConfig.from_env(),
    index_config: IndexConfig = IndexConfig(),
    resume: bool = True,
) -> None:
    """
    Main function to build the FAISS index.
    
    Args:
        model_config: Configuration for the embedding model
        db_config: Database connection configuration
        index_config: Indexing parameters
        resume: Whether to resume from checkpoint if available
    """
    stats = IndexingStats(start_time=time.perf_counter())
    
    # Initialize components
    logger.info("=== Starting Indexing Job ===")
    
    try:
        # Load model
        embedding_model = EmbeddingModel(model_config)
        
        # Connect to database
        db = DatabaseReader(db_config)
        db.connect()
        
        # Initialize FAISS indexer
        indexer = FAISSIndexer(
            embedding_model.embedding_dim, 
            index_config,
            model_config.name
        )
        
        # Try to resume from checkpoint
        offset = 0
        if resume:
            checkpoint_offset = indexer.load_checkpoint()
            if checkpoint_offset is not None:
                offset = checkpoint_offset
                logger.info(f"Resuming from offset {offset}")
        
        # Main indexing loop
        while True:
            try:
                # Fetch batch from database
                rows = db.fetch_batch(index_config.db_batch_size, offset)
                
                if not rows:
                    logger.info("No more data to process")
                    break
                
                doc_ids = [row[0] for row in rows]
                texts = [row[1] for row in rows]
                
                # Embed texts
                embeddings = embedding_model.encode(
                    texts,
                    batch_size=index_config.embed_batch_size
                )
                
                # Add to index
                indexer.add_embeddings(embeddings, doc_ids)
                
                # Update stats
                stats.total_documents += len(texts)
                stats.total_batches += 1
                offset += index_config.db_batch_size
                
                # Log progress
                throughput = stats.get_throughput()
                logger.info(
                    f"Batch {stats.total_batches}: "
                    f"{len(texts)} docs | "
                    f"Total: {stats.total_documents} | "
                    f"Throughput: {throughput:.2f} docs/sec"
                )
                
                # Save checkpoint periodically
                if stats.total_documents % index_config.checkpoint_interval == 0:
                    indexer.save_checkpoint(offset)
                
            except Exception as e:
                logger.error(f"Error processing batch at offset {offset}: {e}")
                stats.errors += 1
                if stats.errors > 10:
                    logger.error("Too many errors, aborting")
                    raise
                # Continue to next batch
                offset += index_config.db_batch_size
                continue
        
        # Save final index
        logger.info("=== Saving Final Index ===")
        indexer.save()
        
        # Final statistics
        elapsed = time.perf_counter() - stats.start_time
        logger.info("=== Indexing Complete ===")
        logger.info(f"Total documents indexed: {stats.total_documents}")
        logger.info(f"Total vectors in index: {indexer.index.ntotal}")
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Average throughput: {stats.get_throughput():.2f} docs/sec")
        logger.info(f"Errors encountered: {stats.errors}")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise
    finally:
        if 'db' in locals():
            db.close()


if __name__ == "__main__":
    # Run with default configuration
    build_index()