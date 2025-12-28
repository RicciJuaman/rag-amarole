"""
Example usage of the RAG system.
Shows common patterns for indexing and retrieval.
"""

import logging
from config import ModelConfig, DatabaseConfig, IndexConfig, RetrievalConfig
from indexer import build_index
from retriever import FAISSRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_build_index():
    """Example 1: Build the FAISS index from database."""
    logger.info("="*60)
    logger.info("EXAMPLE 1: Building Index")
    logger.info("="*60)
    
    # Use default configuration
    build_index()
    
    # Or customize configuration
    custom_config = IndexConfig(
        db_batch_size=1000,        # Larger batches
        embed_batch_size=512,      # More GPU memory
        checkpoint_interval=10000  # Less frequent checkpoints
    )
    
    # build_index(index_config=custom_config)


def example_2_simple_search():
    """Example 2: Simple semantic search."""
    logger.info("="*60)
    logger.info("EXAMPLE 2: Simple Search")
    logger.info("="*60)
    
    # Initialize retriever
    retriever = FAISSRetriever(
        model_config=ModelConfig(),
        index_config=IndexConfig(),
        retrieval_config=RetrievalConfig(top_k=5)
    )
    
    # Search
    query = "What do customers say about the quality?"
    logger.info(f"Query: {query}")
    
    results = retriever.search(query)
    
    logger.info(f"Found {len(results)} results:")
    for result in results:
        logger.info(f"  Rank {result.rank}: Doc {result.doc_id} (score: {result.score:.4f})")


def example_3_batch_search():
    """Example 3: Batch search multiple queries."""
    logger.info("="*60)
    logger.info("EXAMPLE 3: Batch Search")
    logger.info("="*60)
    
    retriever = FAISSRetriever(
        model_config=ModelConfig(),
        index_config=IndexConfig(),
        retrieval_config=RetrievalConfig(top_k=3)
    )
    
    queries = [
        "excellent product quality",
        "shipping delays",
        "customer service issues",
        "value for money"
    ]
    
    logger.info(f"Searching {len(queries)} queries...")
    all_results = retriever.batch_search(queries)
    
    for query, results in zip(queries, all_results):
        logger.info(f"\nQuery: '{query}'")
        for result in results:
            logger.info(f"  - Doc {result.doc_id}: {result.score:.4f}")


def example_4_filtered_search():
    """Example 4: Search with similarity threshold."""
    logger.info("="*60)
    logger.info("EXAMPLE 4: Filtered Search")
    logger.info("="*60)
    
    # Only return results above 0.5 similarity
    retriever = FAISSRetriever(
        model_config=ModelConfig(),
        index_config=IndexConfig(),
        retrieval_config=RetrievalConfig(
            top_k=10,
            min_similarity=0.5
        )
    )
    
    query = "battery life concerns"
    logger.info(f"Query: {query} (min_similarity=0.5)")
    
    results = retriever.search(query)
    
    if results:
        logger.info(f"Found {len(results)} high-quality matches:")
        for result in results:
            logger.info(f"  Doc {result.doc_id}: {result.score:.4f}")
    else:
        logger.info("No results above similarity threshold")


def example_5_custom_model():
    """Example 5: Using a different embedding model."""
    logger.info("="*60)
    logger.info("EXAMPLE 5: Custom Model Configuration")
    logger.info("="*60)
    
    # You can switch to a different model
    # Just make sure to rebuild the index with the same model
    custom_model_config = ModelConfig(
        name="sentence-transformers/all-mpnet-base-v2",
        max_seq_length=384,  # Shorter sequences
        dtype="float32",      # Higher precision
        device="cuda"         # Force GPU
    )
    
    logger.info(f"Model: {custom_model_config.name}")
    logger.info(f"Max sequence length: {custom_model_config.max_seq_length}")
    logger.info(f"Data type: {custom_model_config.dtype}")
    
    # Build index with custom model
    # build_index(model_config=custom_model_config)
    
    # Search with custom model
    retriever = FAISSRetriever(
        model_config=custom_model_config,
        index_config=IndexConfig(),
        retrieval_config=RetrievalConfig()
    )
    
    results = retriever.search("test query", top_k=3)
    logger.info(f"Retrieved {len(results)} results")


def example_6_production_setup():
    """Example 6: Production-ready configuration."""
    logger.info("="*60)
    logger.info("EXAMPLE 6: Production Setup")
    logger.info("="*60)
    
    # Optimized configuration for production
    prod_model_config = ModelConfig(
        name="sentence-transformers/all-mpnet-base-v2",
        max_seq_length=512,
        dtype="float16",  # Balance speed/memory
        device=None       # Auto-detect
    )
    
    prod_index_config = IndexConfig(
        db_batch_size=1000,
        embed_batch_size=512,
        checkpoint_interval=10000
    )
    
    prod_retrieval_config = RetrievalConfig(
        top_k=10,
        min_similarity=0.3,  # Filter low-quality matches
        use_bm25=False       # Can enable for hybrid
    )
    
    logger.info("Production configuration:")
    logger.info(f"  Model: {prod_model_config.name}")
    logger.info(f"  DB batch: {prod_index_config.db_batch_size}")
    logger.info(f"  Embed batch: {prod_index_config.embed_batch_size}")
    logger.info(f"  Top-k: {prod_retrieval_config.top_k}")
    
    # Build and search
    # build_index(
    #     model_config=prod_model_config,
    #     index_config=prod_index_config
    # )
    
    # retriever = FAISSRetriever(
    #     model_config=prod_model_config,
    #     index_config=prod_index_config,
    #     retrieval_config=prod_retrieval_config
    # )


def main():
    """Run all examples (commented out by default)."""
    
    # Uncomment the examples you want to run:
    
    # Build index first (only need to do this once)
    # example_1_build_index()
    
    # Search examples (require index to be built first)
    example_2_simple_search()
    # example_3_batch_search()
    # example_4_filtered_search()
    # example_5_custom_model()
    # example_6_production_setup()


if __name__ == "__main__":
    main()