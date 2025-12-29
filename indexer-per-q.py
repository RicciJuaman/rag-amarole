import os
import psycopg2 # pyright: ignore[reportMissingModuleSource]
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
from typing import List, Tuple
import pickle

class EmbeddingIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding indexer with a specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        load_dotenv()
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.id_map = []  # Maps index position to document ID
        
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
    
    def build_index(self, embeddings: np.ndarray, ids: List[int]):
        """
        Build FAISS index with the embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            ids: List of document IDs
        """
        print(f"Building FAISS index with dimension {self.embedding_dim}...")
        
        # Create FAISS index using Inner Product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store ID mapping
        self.id_map = ids
        
        print(f"Index built successfully with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata (id_map)
        """
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.id_map, f)
        
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
            self.id_map = pickle.load(f)
        
        print(f"Index loaded from {index_path}")
        print(f"Metadata loaded from {metadata_path}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the index for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of tuples (document_id, similarity_score)
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Please build or load an index first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Map indices to document IDs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.id_map):
                results.append((self.id_map[idx], float(dist)))
        
        return results
    
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
        
        # Build index
        self.build_index(embeddings, ids)
        
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
            'labels': dict(zip(ids, labels))
        }
        
        print(f"\nStatistics:")
        print(f"Total documents: {stats['total_docs']}")
        print(f"Relevant (YES): {stats['relevant_docs']}")
        print(f"Irrelevant (NO): {stats['irrelevant_docs']}")
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize the indexer with your chosen model
    indexer = EmbeddingIndexer(model_name='all-MiniLM-L6-v2')
    
    # Example: Embed query 1 (q1 table)
    stats = indexer.embed_table(
        gold_set_table='q1',
        reviews_table='reviews',
        query_name='Good Dog Food',
        save_index_path='q1_index.faiss',
        save_metadata_path='q1_metadata.pkl'
    )
    
    # Search example
    query = "Good Dog Food"
    results = indexer.search(query, k=10)
    
    print(f"\nTop 10 results for query '{query}':")
    for rank, (doc_id, score) in enumerate(results, 1):
        label = stats['labels'].get(doc_id, 'UNKNOWN')
        print(f"{rank}. ID: {doc_id}, Score: {score:.4f}, Label: {label}")
    
    # Switch to another table (q2, q3, etc.)
    # indexer2 = EmbeddingIndexer(model_name='all-MiniLM-L6-v2')
    # stats2 = indexer2.embed_table(
    #     gold_set_table='q2',
    #     reviews_table='reviews',
    #     query_name='Another Query Name',
    #     save_index_path='q2_index.faiss',
    #     save_metadata_path='q2_metadata.pkl'
    # )