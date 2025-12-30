import os
import psycopg2  # pyright: ignore[reportMissingModuleSource]
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
from typing import List, Tuple, Optional
import pickle
from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingImports]


class EmbeddingIndexer:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', use_bm25: bool = True, bm25_weight: float = 0.3):
        """
        Initialize the embedding indexer with a specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            use_bm25: Whether to enable BM25 hybrid search
            bm25_weight: Weight for BM25 in hybrid mode (0-1, default 0.3)
        """
        load_dotenv()
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.id_map = []  # Maps index position to document ID
        self.texts = []  # Store document texts for BM25
        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self.bm25 = None
        
        # Database connection parameters
        self.db_params = {
            'host': os.getenv('host'),
            'port': os.getenv('port'),
            'dbname': os.getenv('dbname'),
            'user': os.getenv('user'),
            'password': os.getenv('password')
        }
    
    def get_db_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(**self.db_params)
    
    def fetch_gold_set_data(self, gold_set_table: str, reviews_table: str, query_name: str) -> List[Tuple]:
        """
        Fetch gold set data joined with reviews table.
        
        Args:
            gold_set_table: Name of the gold set table (e.g., 'q1', 'q2')
            reviews_table: Name of the reviews table
            query_name: The query name/column to filter by (e.g., 'Good Dog Food')
            
        Returns:
            List of tuples containing (id, summary, text, label)
        """
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        query = f"""
            SELECT 
                g."Id",
                r."Summary",
                r."Text",
                g."{query_name}"
            FROM {gold_set_table} g
            JOIN {reviews_table} r ON g."Id" = r."Id"
            WHERE g."{query_name}" IS NOT NULL
            ORDER BY g."Id"
        """
        
        cur.execute(query)
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return results
    
    def prepare_documents(self, data: List[Tuple]) -> Tuple[List[str], List[int], List[str]]:
        """
        Prepare documents by concatenating summary and text.
        
        Args:
            data: List of tuples (id, summary, text, label)
            
        Returns:
            Tuple of (concatenated_texts, ids, labels)
        """
        texts = []
        ids = []
        labels = []
        
        for row in data:
            doc_id, summary, text, label = row
            
            # Handle None values
            summary = summary or ""
            text = text or ""
            
            # Concatenate summary and text
            combined_text = f"{summary} {text}".strip()
            
            texts.append(combined_text)
            ids.append(doc_id)
            labels.append(label)
        
        return texts, ids, labels
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for the given texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize embeddings for inner product similarity
        faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, ids: List[int], texts: List[str]):
        """
        Build FAISS index with the embeddings and optionally BM25 index.
        
        Args:
            embeddings: Numpy array of embeddings
            ids: List of document IDs
            texts: List of document texts (for BM25)
        """
        print(f"Building FAISS index with dimension {self.embedding_dim}...")
        
        # Create FAISS index using Inner Product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store ID mapping and texts
        self.id_map = ids
        self.texts = texts
        
        # Build BM25 index if enabled
        if self.use_bm25:
            print("Building BM25 index...")
            tokenized_docs = [doc.lower().split() for doc in texts]
            self.bm25 = BM25Okapi(tokenized_docs)
            print(f"BM25 index ready (weight: {self.bm25_weight})")
        
        print(f"Index built successfully with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata (id_map and texts)
        """
        faiss.write_index(self.index, index_path)
        
        metadata = {
            'id_map': self.id_map,
            'texts': self.texts,
            'use_bm25': self.use_bm25,
            'bm25_weight': self.bm25_weight
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load a FAISS index and metadata from disk.
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the metadata file
        """
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.id_map = metadata['id_map']
        self.texts = metadata['texts']
        self.use_bm25 = metadata.get('use_bm25', False)
        self.bm25_weight = metadata.get('bm25_weight', 0.3)
        
        # Rebuild BM25 index if enabled
        if self.use_bm25:
            print("Rebuilding BM25 index...")
            tokenized_docs = [doc.lower().split() for doc in self.texts]
            self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"Index loaded from {index_path}")
        print(f"Metadata loaded from {metadata_path}")
        print(f"BM25 enabled: {self.use_bm25}")
    
    def search(self, query: str, k: int = 10, use_bm25: Optional[bool] = None) -> List[Tuple[int, float, str]]:
        """
        Search the index for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            use_bm25: Override instance setting for this search
            
        Returns:
            List of tuples (document_id, similarity_score, search_type)
            where search_type is 'semantic', 'bm25', or 'hybrid'
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Please build or load an index first.")
        
        # Determine if we should use BM25
        should_use_bm25 = use_bm25 if use_bm25 is not None else self.use_bm25
        
        if should_use_bm25 and self.bm25:
            return self._hybrid_search(query, k)
        else:
            return self._semantic_search(query, k)
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float, str]]:
        """Pure semantic search using FAISS."""
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Map indices to document IDs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.id_map):
                results.append((self.id_map[idx], float(dist), 'semantic'))
        
        return results
    
    def _hybrid_search(self, query: str, k: int) -> List[Tuple[int, float, str]]:
        """Hybrid search combining FAISS and BM25."""
        # Get more results from semantic for reranking
        semantic_k = min(k * 3, len(self.id_map))
        
        # Semantic search
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        semantic_distances, semantic_indices = self.index.search(query_embedding.astype('float32'), semantic_k)
        
        # BM25 search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = bm25_scores / max_bm25
        
        # Combine scores
        semantic_weight = 1.0 - self.bm25_weight
        bm25_weight = self.bm25_weight
        
        combined_scores = {}
        for idx, semantic_score in zip(semantic_indices[0], semantic_distances[0]):
            doc_id = self.id_map[idx]
            bm25_score = bm25_scores_norm[idx]
            
            # Weighted combination
            combined_score = (semantic_weight * semantic_score + bm25_weight * bm25_score)
            combined_scores[doc_id] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Return with 'hybrid' tag
        return [(doc_id, score, 'hybrid') for doc_id, score in sorted_results]
    
    def embed_table(self, gold_set_table: str, reviews_table: str, query_name: str,
                    save_index_path: str = None, save_metadata_path: str = None) -> dict:
        """
        Complete pipeline to embed a table and create index.
        
        Args:
            gold_set_table: Name of the gold set table
            reviews_table: Name of the reviews table
            query_name: Query column name
            save_index_path: Optional path to save the index
            save_metadata_path: Optional path to save the metadata
            
        Returns:
            Dictionary with statistics and labels
        """
        # Fetch data
        print(f"Fetching data from {gold_set_table}...")
        data = self.fetch_gold_set_data(gold_set_table, reviews_table, query_name)
        
        if not data:
            raise ValueError(f"No data found for query '{query_name}' in table '{gold_set_table}'")
        
        # Prepare documents
        texts, ids, labels = self.prepare_documents(data)
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Build index (with BM25 if enabled)
        self.build_index(embeddings, ids, texts)
        
        # Save if paths provided
        if save_index_path and save_metadata_path:
            self.save_index(save_index_path, save_metadata_path)
        
        # Return statistics
        relevant_count = sum(1 for label in labels if label == 'YES')
        irrelevant_count = sum(1 for label in labels if label == 'NO')
        
        stats = {
            'total_docs': len(data),
            'relevant_docs': relevant_count,
            'irrelevant_docs': irrelevant_count,
            'labels': dict(zip(ids, labels)),
            'use_bm25': self.use_bm25,
            'bm25_weight': self.bm25_weight if self.use_bm25 else None
        }
        
        print(f"\nStatistics:")
        print(f"Total documents: {stats['total_docs']}")
        print(f"Relevant (YES): {stats['relevant_docs']}")
        print(f"Irrelevant (NO): {stats['irrelevant_docs']}")
        print(f"BM25 enabled: {stats['use_bm25']}")
        if stats['use_bm25']:
            print(f"BM25 weight: {stats['bm25_weight']}")
        
        return stats
    
    def evaluate_top_k(self, query: str, stats: dict, k: int = 10, use_bm25: Optional[bool] = None):
        """
        Evaluate and display top-k results.
        
        Args:
            query: Query text
            stats: Statistics dictionary from embed_table
            k: Number of results to show
            use_bm25: Override instance setting for this search
        """
        print(f"\n{'='*80}")
        search_type = 'Hybrid (Semantic + BM25)' if (use_bm25 if use_bm25 is not None else self.use_bm25) else 'Semantic Only'
        print(f"Top {k} Results - {search_type}")
        print(f"Query: '{query}'")
        print(f"{'='*80}\n")
        
        results = self.search(query, k=k, use_bm25=use_bm25)
        
        relevant_in_top_k = 0
        
        for rank, (doc_id, score, method) in enumerate(results, 1):
            label = stats['labels'].get(doc_id, 'UNKNOWN')
            is_relevant = '✓' if label == 'YES' else '✗'
            
            if label == 'YES':
                relevant_in_top_k += 1
            
            print(f"{rank:2d}. {is_relevant} ID: {doc_id:8d} | Score: {score:.4f} | Label: {label:3s} | Method: {method}")
        
        # Calculate precision@k
        precision_at_k = relevant_in_top_k / k if k > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"Relevant in top-{k}: {relevant_in_top_k}/{k}")
        print(f"Precision@{k}: {precision_at_k:.2%}")
        print(f"{'='*80}\n")
        
        return {
            'relevant_in_top_k': relevant_in_top_k,
            'precision_at_k': precision_at_k,
            'results': results
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Embedding Indexer with BM25 Support")
    print("="*80)
    
    # Initialize with BM25 enabled (default)
    print("\n### Creating indexer with BM25 enabled ###")
    indexer = EmbeddingIndexer(
        model_name='sentence-transformers/all-mpnet-base-v2',
        use_bm25=True,
        bm25_weight=0.3
    )
    
    # Embed query 1 (q1 table)
    stats = indexer.embed_table(
        gold_set_table='q1',
        reviews_table='reviews',
        query_name='Good Dog Food',
        save_index_path='q1_index.faiss',
        save_metadata_path='q1_metadata.pkl'
    )
    
    query = "Good Dog Food"
    
    # Show top 10 with semantic only
    print("\n" + "="*80)
    print("COMPARISON: Semantic vs Hybrid Search")
    print("="*80)
    
    results_semantic = indexer.evaluate_top_k(query, stats, k=10, use_bm25=False)
    results_hybrid = indexer.evaluate_top_k(query, stats, k=10, use_bm25=True)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"\nSemantic Only:")
    print(f"  Precision@10: {results_semantic['precision_at_k']:.2%}")
    print(f"  Relevant docs: {results_semantic['relevant_in_top_k']}/10")
    
    print(f"\nHybrid (Semantic + BM25):")
    print(f"  Precision@10: {results_hybrid['precision_at_k']:.2%}")
    print(f"  Relevant docs: {results_hybrid['relevant_in_top_k']}/10")
    
    improvement = results_hybrid['precision_at_k'] - results_semantic['precision_at_k']
    print(f"\nImprovement: {improvement:+.2%}")
    
    # Example: Load and search later
    # indexer2 = EmbeddingIndexer(use_bm25=True)
    # indexer2.load_index('q1_index.faiss', 'q1_metadata.pkl')
    # results = indexer2.search("Good Dog Food", k=10)