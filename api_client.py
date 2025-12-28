"""
Example client for the RAG Search API.

Usage:
    python api_client.py
"""

import requests
from typing import List, Optional
import json


class RAGClient:
    """Client for interacting with the RAG Search API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> dict:
        """Check if the API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        min_score: Optional[float] = None
    ) -> dict:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            Dictionary with search results including metadata
        """
        payload = {
            "query": query,
            "top_k": top_k
        }
        if min_score is not None:
            payload["min_score"] = min_score
        
        response = requests.post(
            f"{self.base_url}/search",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def search_get(
        self, 
        query: str, 
        top_k: int = 10, 
        min_score: Optional[float] = None
    ) -> dict:
        """
        Search using GET method (alternative).
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            Dictionary with search results including metadata
        """
        params = {
            "q": query,
            "top_k": top_k
        }
        if min_score is not None:
            params["min_score"] = min_score
        
        response = requests.get(
            f"{self.base_url}/search",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_document(self, doc_id: int) -> dict:
        """
        Get a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary with document metadata
        """
        response = requests.get(f"{self.base_url}/document/{doc_id}")
        response.raise_for_status()
        return response.json()


def print_results(results: dict, show_full_text: bool = False):
    """Pretty print search results."""
    print(f"\n{'='*80}")
    print(f"Query: {results['query']}")
    print(f"Total Results: {results['total_results']}")
    print(f"{'='*80}\n")
    
    for result in results['results']:
        print(f"Rank {result['rank']}: Doc ID {result['doc_id']} (Score: {result['score']:.4f})")
        print(f"Summary: {result['summary'][:150]}..." if len(result['summary']) > 150 else f"Summary: {result['summary']}")
        
        if show_full_text:
            print(f"\nFull Text:\n{result['text']}\n")
        else:
            print(f"Text Preview: {result['text'][:200]}..." if len(result['text']) > 200 else f"Text: {result['text']}")
        
        print("-" * 80)


def example_1_basic_search():
    """Example 1: Basic search."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Search")
    print("="*80)
    
    client = RAGClient()
    
    # Check health
    health = client.health_check()
    print(f"✓ API Status: {health['status']}")
    print(f"✓ Index Size: {health['index_size']:,} documents\n")
    
    # Perform search
    query = "What do customers say about product quality?"
    results = client.search(query, top_k=3)
    
    print_results(results)


def example_2_filtered_search():
    """Example 2: Search with score filter."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Filtered Search (min_score=0.5)")
    print("="*80)
    
    client = RAGClient()
    
    query = "shipping and delivery issues"
    results = client.search(query, top_k=5, min_score=0.5)
    
    print_results(results)


def example_3_get_document():
    """Example 3: Get specific document."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Get Specific Document")
    print("="*80)
    
    client = RAGClient()
    
    # First, search to get a doc_id
    results = client.search("great product", top_k=1)
    
    if results['results']:
        doc_id = results['results'][0]['doc_id']
        print(f"\nFetching document {doc_id}...\n")
        
        # Get the full document
        doc = client.get_document(doc_id)
        
        print(f"Document ID: {doc['doc_id']}")
        print(f"Summary: {doc['summary']}")
        print(f"\nFull Text:\n{doc['text']}")


def example_4_batch_queries():
    """Example 4: Multiple queries."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Queries")
    print("="*80)
    
    client = RAGClient()
    
    queries = [
        "product quality",
        "customer service",
        "value for money",
        "shipping speed"
    ]
    
    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        results = client.search(query, top_k=2)
        
        for result in results['results']:
            print(f"  {result['rank']}. Doc {result['doc_id']} (Score: {result['score']:.4f})")
            print(f"     Summary: {result['summary'][:80]}...")


def example_5_stats():
    """Example 5: Get system statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 5: System Statistics")
    print("="*80)
    
    client = RAGClient()
    
    stats = client.get_stats()
    
    print(f"\nSystem Statistics:")
    print(f"  Total Documents: {stats['total_documents']:,}")
    print(f"  Index Type: {stats['index_type']}")
    print(f"  Embedding Model: {stats['embedding_model']}")
    print(f"  Embedding Dimension: {stats['embedding_dimension']}")


def example_6_error_handling():
    """Example 6: Error handling."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Error Handling")
    print("="*80)
    
    client = RAGClient()
    
    # Try to get non-existent document
    try:
        doc = client.get_document(999999999)
        print(doc)
    except requests.exceptions.HTTPError as e:
        print(f"✓ Correctly caught error: {e.response.status_code} - {e.response.json()}")
    
    # Try invalid query
    try:
        results = client.search("", top_k=5)
    except requests.exceptions.HTTPError as e:
        print(f"✓ Correctly caught error: {e.response.status_code} - {e.response.json()}")


def interactive_search():
    """Interactive search mode."""
    print("="*80)
    print("RAG API - Interactive Search")
    print("="*80)
    
    client = RAGClient()
    
    # Check connection
    try:
        health = client.health_check()
        print(f"✓ Connected to API ({health['index_size']:,} documents)")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure the API is running:")
        print("  uvicorn api:app --reload")
        return
    
    print("\nCommands:")
    print("  - Type your query and press Enter")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'stats' for system statistics")
    print()
    
    while True:
        try:
            query = input("Search> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'stats':
                stats = client.get_stats()
                print(f"\n  Documents: {stats['total_documents']:,}")
                print(f"  Model: {stats['embedding_model']}")
                print()
                continue
            
            # Perform search
            results = client.search(query, top_k=3)
            
            print(f"\n  Found {results['total_results']} results:")
            for result in results['results']:
                print(f"\n  Rank {result['rank']}: Doc {result['doc_id']} (Score: {result['score']:.4f})")
                print(f"  Summary: {result['summary'][:100]}...")
                print(f"  Text: {result['text'][:150]}...")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}\n")


def main():
    """Run examples."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_search()
    else:
        print("\n" + "="*80)
        print("RAG API Client Examples")
        print("="*80)
        print("\nMake sure the API is running first:")
        print("  uvicorn api:app --reload")
        print("\nOr run in interactive mode:")
        print("  python api_client.py interactive")
        
        # Run examples
        try:
            example_1_basic_search()
            # example_2_filtered_search()
            # example_3_get_document()
            # example_4_batch_queries()
            # example_5_stats()
            # example_6_error_handling()
        except requests.exceptions.ConnectionError:
            print("\n✗ Could not connect to API. Make sure it's running:")
            print("  uvicorn api:app --reload")
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()