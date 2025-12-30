"""
Search and retrieval functionality for the RAG system.
Supports semantic search using FAISS and hybrid retrieval with BM25.
"""

import logging
import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import faiss # pyright: ignore[reportMissingImports]
from rank_bm25 import BM25Okapi # pyright: ignore[reportMissingImports]

from config import ModelConfig, IndexConfig, RetrievalConfig
from indexer import EmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for a single search result."""
    doc_id: int
    score: float
    rank: int
    
    def __repr__(self):
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, rank={self.rank})"


class FAISSRetriever:
    """Handle loading and searching the FAISS index."""
    
    def __init__(
        self, 
        model_config: ModelConfig,
        index_config: IndexConfig,
        retrieval_config: RetrievalConfig
    ):
        self.model_config = model_config
        self.index_config = index_config
        self.retrieval_config = retrieval_config
        
        # Load model
        self.embedding_model = EmbeddingModel(model_config)
        
        # Load index and metadata
        self.index, self.doc_ids = self._load_index()
        
        logger.info(f"Retriever ready with {len(self.doc_ids)} documents")
    
    def _load_index(self) -> Tuple[faiss.Index, List[int], Optional[List[str]]]:
        """Load FAISS index, document IDs, and optionally document texts."""
        index_path = self.index_config.get_index_path(self.model_config.name)
        meta_path = self.index_config.get_metadata_path(self.model_config.name)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}")
        
        try:
            logger.info(f"Loading FAISS index from {index_path}")
            index = faiss.read_index(str(index_path))
            
            logger.info(f"Loading metadata from {meta_path}")
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            
            # Handle both old format (just doc_ids) and new format (dict with doc_ids and doc_texts)
            if isinstance(metadata, dict):
                doc_ids = metadata['doc_ids']
                doc_texts = metadata.get('doc_texts', None)
            else:
                # Old format - just a list of doc_ids
                doc_ids = metadata
                doc_texts = None
                logger.warning("Old metadata format detected (no texts for BM25)")
            
            if index.ntotal != len(doc_ids):
                raise ValueError(
                    f"Index size mismatch: {index.ntotal} vectors but {len(doc_ids)} doc IDs"
                )
            
            if doc_texts and len(doc_texts) != len(doc_ids):
                logger.warning(f"Text count mismatch: {len(doc_texts)} texts but {len(doc_ids)} doc IDs")
                doc_texts = None
            
            return index, doc_ids, doc_texts
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def search(self, query: str, top_k: Optional[int] = None, use_bm25: Optional[bool] = None) -> List[SearchResult]:
        """
        Search for similar documents using semantic similarity and optionally BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return (defaults to config value)
            use_bm25: Override config to enable/disable BM25 for this search
            
        Returns:
            List of SearchResult objects, sorted by score (highest first)
        """
        if top_k is None:
            top_k = self.retrieval_config.top_k
        
        # Determine if we should use BM25 for this search
        should_use_bm25 = use_bm25 if use_bm25 is not None else self.retrieval_config.use_bm25
        
        # If BM25 is requested and available, use hybrid search
        if should_use_bm25 and self.bm25:
            return self._hybrid_search(query, top_k)
        else:
            # Pure semantic search
            return self._semantic_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Pure semantic search using FAISS."""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                batch_size=1
            )
            
            # Search index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Convert to SearchResult objects
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
                if score >= self.retrieval_config.min_similarity:
                    results.append(SearchResult(
                        doc_id=self.doc_ids[idx],
                        score=float(score),
                        rank=rank
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _hybrid_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Hybrid search combining FAISS semantic search and BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult with combined scores
        """
        try:
            # Get more results from semantic search for reranking
            semantic_top_k = min(top_k * 3, len(self.doc_ids))
            
            # Encode query for semantic search
            query_embedding = self.embedding_model.encode([query], batch_size=1)
            
            # Get semantic results
            semantic_scores, semantic_indices = self.index.search(query_embedding, semantic_top_k)
            
            # Get BM25 scores for all documents
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Normalize BM25 scores to 0-1 range
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores_norm = bm25_scores / max_bm25
            
            # Combine scores with weighted average
            semantic_weight = 1.0 - self.retrieval_config.bm25_weight
            bm25_weight = self.retrieval_config.bm25_weight
            
            combined_scores = {}
            for idx, semantic_score in zip(semantic_indices[0], semantic_scores[0]):
                doc_id = self.doc_ids[idx]
                bm25_score = bm25_scores_norm[idx]
                
                # Weighted combination
                combined_score = (semantic_weight * semantic_score + 
                                bm25_weight * bm25_score)
                combined_scores[doc_id] = combined_score
            
            # Sort by combined score
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            # Convert to SearchResult objects
            results = []
            for rank, (doc_id, score) in enumerate(sorted_results, start=1):
                if score >= self.retrieval_config.min_similarity:
                    results.append(SearchResult(
                        doc_id=doc_id,
                        score=float(score),
                        rank=rank
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    def batch_search(
        self, 
        queries: List[str], 
        top_k: Optional[int] = None,
        use_bm25: Optional[bool] = None
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries at once.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            use_bm25: Override config to enable/disable BM25
            
        Returns:
            List of result lists, one for each query
        """
        # For batch search with BM25, process queries individually
        should_use_bm25 = use_bm25 if use_bm25 is not None else self.retrieval_config.use_bm25
        
        if should_use_bm25 and self.bm25:
            return [self.search(query, top_k, use_bm25=True) for query in queries]
        
        # Pure semantic batch search
        if top_k is None:
            top_k = self.retrieval_config.top_k
        
        try:
            # Encode all queries
            query_embeddings = self.embedding_model.encode(
                queries,
                batch_size=len(queries)
            )
            
            # Search index
            scores, indices = self.index.search(query_embeddings, top_k)
            
            # Convert to SearchResult objects for each query
            all_results = []
            for query_idx in range(len(queries)):
                results = []
                for rank, (idx, score) in enumerate(
                    zip(indices[query_idx], scores[query_idx]), 
                    start=1
                ):
                    if score >= self.retrieval_config.min_similarity:
                        results.append(SearchResult(
                            doc_id=self.doc_ids[idx],
                            score=float(score),
                            rank=rank
                        ))
                all_results.append(results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            raise


def demo_search():
    """Demo function showing how to use the retriever."""
    
    # Initialize configuration
    model_config = ModelConfig()
    index_config = IndexConfig()
    
    # Demo 1: Pure semantic search
    print("\n" + "="*60)
    print("DEMO 1: Pure Semantic Search (FAISS only)")
    print("="*60)
    
    retrieval_config = RetrievalConfig(top_k=5, use_bm25=False)
    retriever = FAISSRetriever(
        model_config=model_config,
        index_config=index_config,
        retrieval_config=retrieval_config
    )
    
    query = "What are the best features of this product?"
    logger.info(f"Query: {query}")
    results = retriever.search(query)
    
    for result in results:
        logger.info(f"  Rank {result.rank}: Doc ID={result.doc_id}, Score={result.score:.4f}")
    
    # Demo 2: Hybrid search (if BM25 is available)
    if retriever.doc_texts:
        print("\n" + "="*60)
        print("DEMO 2: Hybrid Search (FAISS + BM25)")
        print("="*60)
        
        retrieval_config_hybrid = RetrievalConfig(
            top_k=5, 
            use_bm25=True,
            bm25_weight=0.3
        )
        retriever_hybrid = FAISSRetriever(
            model_config=model_config,
            index_config=index_config,
            retrieval_config=retrieval_config_hybrid
        )
        
        query = "excellent quality and fast shipping"
        logger.info(f"Query: {query}")
        results = retriever_hybrid.search(query)
        
        for result in results:
            logger.info(f"  Rank {result.rank}: Doc ID={result.doc_id}, Score={result.score:.4f}")
        
        # Demo 3: Toggle BM25 per query
        print("\n" + "="*60)
        print("DEMO 3: Toggle BM25 Per Query")
        print("="*60)
        
        query = "customer service"
        
        # Search without BM25
        logger.info(f"Query (semantic only): {query}")
        results_semantic = retriever_hybrid.search(query, use_bm25=False)
        for result in results_semantic[:3]:
            logger.info(f"  Rank {result.rank}: Doc {result.doc_id}, Score={result.score:.4f}")
        
        # Search with BM25
        logger.info(f"\nQuery (with BM25): {query}")
        results_hybrid = retriever_hybrid.search(query, use_bm25=True)
        for result in results_hybrid[:3]:
            logger.info(f"  Rank {result.rank}: Doc {result.doc_id}, Score={result.score:.4f}")
    else:
        logger.warning("\nBM25 not available - index was built without document texts")
        logger.warning("Rebuild the index to enable BM25 hybrid search")


if __name__ == "__main__":
    demo_search()