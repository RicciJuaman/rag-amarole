"""
FastAPI server for RAG system with document metadata retrieval.

Usage:
    uvicorn api:app --reload
    
Or:
    python api.py
"""

import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from config import ModelConfig, IndexConfig, RetrievalConfig, DatabaseConfig
from retriever import FAISSRetriever
from indexer import DatabaseReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
retriever = None
db_reader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global retriever, db_reader
    
    # Startup
    logger.info("Initializing RAG system...")
    try:
        retriever = FAISSRetriever(
            model_config=ModelConfig(),
            index_config=IndexConfig(),
            retrieval_config=RetrievalConfig()
        )
        logger.info(f"✓ Index loaded: {len(retriever.doc_ids):,} documents")
        
        # Initialize database reader for metadata
        db_reader = DatabaseReader(DatabaseConfig.from_env())
        db_reader.connect()
        logger.info("✓ Database connected")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if db_reader:
        db_reader.close()


# Create FastAPI app
app = FastAPI(
    title="RAG Search API",
    description="Semantic search API for product reviews with metadata",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: Optional[int] = Field(10, description="Number of results to return", ge=1, le=100)
    min_score: Optional[float] = Field(None, description="Minimum similarity score (0-1)", ge=0, le=1)


class DocumentMetadata(BaseModel):
    """Document metadata including summary and text."""
    doc_id: int = Field(..., description="Document ID")
    summary: str = Field(..., description="Document summary")
    text: str = Field(..., description="Full document text")
    score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Result rank")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Number of results returned")
    results: List[DocumentMetadata] = Field(..., description="Search results with metadata")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    index_size: int
    message: str


class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_documents: int
    index_type: str
    embedding_model: str
    embedding_dimension: int


# Helper function to fetch metadata
def fetch_document_metadata(doc_ids: List[int]) -> dict:
    """
    Fetch summary and text for given document IDs.
    
    Args:
        doc_ids: List of document IDs
        
    Returns:
        Dictionary mapping doc_id to (summary, text)
    """
    if not doc_ids:
        return {}
    
    # Build query with parameterized IN clause
    placeholders = ','.join(['%s'] * len(doc_ids))
    query = f"""
        SELECT "Id", 
               COALESCE("Summary", '') AS summary,
               COALESCE("Text", '') AS text
        FROM reviews
        WHERE "Id" IN ({placeholders})
    """
    
    try:
        with db_reader.conn.cursor() as cur:
            cur.execute(query, doc_ids)
            rows = cur.fetchall()
        
        # Create dictionary mapping doc_id -> (summary, text)
        return {row[0]: (row[1], row[2]) for row in rows}
        
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch document metadata")


# API Endpoints

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return HealthResponse(
        status="healthy",
        index_size=len(retriever.doc_ids),
        message="RAG system is operational"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return StatsResponse(
        total_documents=len(retriever.doc_ids),
        index_type="FAISS IndexFlatIP",
        embedding_model=retriever.model_config.name,
        embedding_dimension=retriever.embedding_model.embedding_dim
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for similar documents.
    
    Returns documents with full metadata (summary and text).
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Update retrieval config if needed
        if request.min_score is not None:
            retriever.retrieval_config.min_similarity = request.min_score
        
        # Perform search
        logger.info(f"Search query: '{request.query}' (top_k={request.top_k})")
        results = retriever.search(request.query, top_k=request.top_k)
        
        if not results:
            return SearchResponse(
                query=request.query,
                total_results=0,
                results=[]
            )
        
        # Fetch metadata for all results
        doc_ids = [r.doc_id for r in results]
        metadata_map = fetch_document_metadata(doc_ids)
        
        # Build response with metadata
        results_with_metadata = []
        for result in results:
            if result.doc_id in metadata_map:
                summary, text = metadata_map[result.doc_id]
                results_with_metadata.append(DocumentMetadata(
                    doc_id=result.doc_id,
                    summary=summary,
                    text=text,
                    score=result.score,
                    rank=result.rank
                ))
            else:
                logger.warning(f"Metadata not found for doc_id: {result.doc_id}")
        
        logger.info(f"Returning {len(results_with_metadata)} results")
        
        return SearchResponse(
            query=request.query,
            total_results=len(results_with_metadata),
            results=results_with_metadata
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(10, description="Number of results", ge=1, le=100),
    min_score: Optional[float] = Query(None, description="Minimum score", ge=0, le=1)
):
    """
    Search endpoint using GET method (alternative to POST).
    
    Usage: /search?q=your+query&top_k=5&min_score=0.5
    """
    request = SearchRequest(query=q, top_k=top_k, min_score=min_score)
    return await search(request)


@app.get("/document/{doc_id}", response_model=DocumentMetadata)
async def get_document(doc_id: int):
    """
    Get a specific document by ID.
    
    Returns the document metadata without similarity score.
    """
    if db_reader is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        metadata_map = fetch_document_metadata([doc_id])
        
        if doc_id not in metadata_map:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        summary, text = metadata_map[doc_id]
        
        return DocumentMetadata(
            doc_id=doc_id,
            summary=summary,
            text=text,
            score=1.0,  # Not applicable for direct fetch
            rank=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "path": str(request.url)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}


def main():
    """Run the server."""
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()