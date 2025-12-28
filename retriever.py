"""
Search and retrieval functionality for the RAG system.
Supports semantic search using FAISS and optional BM25 hybrid retrieval.
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
    
    def _load_index(self) -> Tuple[faiss.Index, List[int]]:
        """Load FAISS index and document ID mapping."""
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
                doc_ids = pickle.load(f)
            
            if index.ntotal != len(doc_ids):
                raise ValueError(
                    f"Index size mismatch: {index.ntotal} vectors but {len(doc_ids)} doc IDs"
                )
            
            return index, doc_ids
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return (defaults to config value)
            
        Returns:
            List of SearchResult objects, sorted by score (highest first)
        """
        if top_k is None:
            top_k = self.retrieval_config.top_k
        
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
    
    def batch_search(
        self, 
        queries: List[str], 
        top_k: Optional[int] = None
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries at once.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            
        Returns:
            List of result lists, one for each query
        """
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


class HybridRetriever:
    """
    Hybrid retrieval combining semantic (FAISS) and lexical (BM25) search.
    
    Note: This requires storing document texts for BM25.
    For now, this is a placeholder showing the architecture.
    """
    
    def __init__(
        self,
        faiss_retriever: FAISSRetriever,
        document_texts: Optional[List[str]] = None
    ):
        self.faiss_retriever = faiss_retriever
        self.config = faiss_retriever.retrieval_config
        
        # Initialize BM25 if texts are provided and hybrid mode is enabled
        self.bm25 = None
        if self.config.use_bm25 and document_texts:
            logger.info("Initializing BM25 index...")
            tokenized_docs = [doc.lower().split() for doc in document_texts]
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info("BM25 index ready")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Hybrid search combining FAISS and BM25 scores.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects with combined scores
        """
        if not self.config.use_bm25 or self.bm25 is None:
            # Fall back to pure semantic search
            return self.faiss_retriever.search(query, top_k)
        
        if top_k is None:
            top_k = self.config.top_k
        
        # Get semantic results
        semantic_results = self.faiss_retriever.search(query, top_k * 2)
        
        # Get BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Combine scores (weighted average)
        combined_results = {}
        semantic_weight = 1.0 - self.config.bm25_weight
        
        for result in semantic_results:
            idx = self.faiss_retriever.doc_ids.index(result.doc_id)
            semantic_score = result.score
            bm25_score = bm25_scores[idx]
            
            # Normalize and combine
            combined_score = (
                semantic_weight * semantic_score + 
                self.config.bm25_weight * (bm25_score / (bm25_score + 1))
            )
            combined_results[result.doc_id] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_results.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Convert to SearchResult objects
        return [
            SearchResult(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(sorted_results, start=1)
        ]


def demo_search():
    """Demo function showing how to use the retriever."""
    
    # Initialize configuration
    model_config = ModelConfig()
    index_config = IndexConfig()
    retrieval_config = RetrievalConfig(top_k=5)
    
    # Create retriever
    logger.info("Initializing retriever...")
    retriever = FAISSRetriever(
        model_config=model_config,
        index_config=index_config,
        retrieval_config=retrieval_config
    )
    
    # Example searches
    queries = [
        "What are the best features of this product?",
        "Any complaints about quality?",
        "Customer service experience"
    ]
    
    for query in queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}")
        
        results = retriever.search(query, top_k=3)
        
        if not results:
            logger.info("No results found")
            continue
        
        for result in results:
            logger.info(f"  Rank {result.rank}: Doc ID={result.doc_id}, Score={result.score:.4f}")


if __name__ == "__main__":
    demo_search()