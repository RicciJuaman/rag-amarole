#!/usr/bin/env python
"""
Simple CLI tool for searching the RAG index.

Usage:
    python search.py "your search query"
    python search.py "your query" --top-k 10
    python search.py "your query" --min-score 0.5
"""

import argparse
import logging
import sys
from typing import Optional

from config import ModelConfig, IndexConfig, RetrievalConfig
from retriever import FAISSRetriever

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings/errors by default
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def search_interactive():
    """Interactive search mode."""
    print("="*60)
    print("RAG Search - Interactive Mode")
    print("="*60)
    print("Loading index...")
    
    try:
        retriever = FAISSRetriever(
            model_config=ModelConfig(),
            index_config=IndexConfig(),
            retrieval_config=RetrievalConfig()
        )
        print(f"✓ Ready! Index has {len(retriever.doc_ids):,} documents")
        print("\nCommands:")
        print("  - Type your query and press Enter")
        print("  - 'quit' or 'exit' to exit")
        print("  - 'help' for options")
        print()
        
        while True:
            try:
                query = input("Search> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\nOptions:")
                    print("  Search: Just type your query")
                    print("  Exit: 'quit' or 'exit'")
                    print()
                    continue
                
                # Perform search
                results = retriever.search(query, top_k=5)
                
                if not results:
                    print("  No results found\n")
                    continue
                
                print(f"\n  Found {len(results)} results:")
                for result in results:
                    print(f"    {result.rank}. Doc {result.doc_id:8d} - Score: {result.score:.4f}")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Search error: {e}")
                
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run 'python indexer.py' first to build the index.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)


def search_single(
    query: str,
    top_k: int = 10,
    min_score: Optional[float] = None,
    verbose: bool = False
):
    """Single query search."""
    
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Loading index...")
    
    try:
        retrieval_config = RetrievalConfig(top_k=top_k)
        if min_score is not None:
            retrieval_config.min_similarity = min_score
        
        retriever = FAISSRetriever(
            model_config=ModelConfig(),
            index_config=IndexConfig(),
            retrieval_config=retrieval_config
        )
        
        if verbose:
            logger.info(f"Index loaded: {len(retriever.doc_ids):,} documents")
            logger.info(f"Searching for: '{query}'")
        
        results = retriever.search(query)
        
        if not results:
            print("No results found")
            return
        
        # Output results
        print(f"Query: {query}")
        print(f"Results: {len(results)}/{top_k}")
        print("-" * 60)
        
        for result in results:
            print(f"Rank {result.rank:2d}: Doc {result.doc_id:8d} | Score: {result.score:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python indexer.py' first to build the index.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search the RAG index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python search.py
  
  # Single query
  python search.py "product quality issues"
  
  # More results
  python search.py "shipping delays" --top-k 10
  
  # Filter by score
  python search.py "customer service" --min-score 0.5
  
  # Verbose output
  python search.py "battery life" --verbose
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (omit for interactive mode)'
    )
    
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    
    parser.add_argument(
        '-s', '--min-score',
        type=float,
        help='Minimum similarity score (0-1)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Force interactive mode'
    )
    
    args = parser.parse_args()
    
    # Decide mode
    if args.query and not args.interactive:
        # Single query mode
        search_single(
            query=args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            verbose=args.verbose
        )
    else:
        # Interactive mode
        if args.query:
            print("Note: Ignoring query argument in interactive mode")
        search_interactive()


if __name__ == "__main__":
    main()