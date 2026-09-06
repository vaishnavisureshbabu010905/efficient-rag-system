"""
API request/response schemas for M6.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request schema for RAG query."""
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    
    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        example = {
            "query": "What is machine learning?",
            "top_k": 5
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status (healthy/degraded)")
    version: str = Field(..., description="API version")
    
    class Config:
        example = {
            "status": "healthy",
            "version": "1.0.0"
        }


class RetrievalMetadata(BaseModel):
    """Metadata about a retrieved chunk."""
    chunk_id: str
    fusion_score: Optional[float] = None
    rank: Optional[int] = None


class QueryResponse(BaseModel):
    """Response schema for RAG query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default=[], description="List of chunk IDs used")
    has_sufficient_context: bool = Field(..., description="Whether sufficient context was found")
    num_chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    retrieval_metadata: List[RetrievalMetadata] = Field(default=[], description="Details of retrieved chunks")
    latency_ms: Optional[float] = Field(None, description="Query latency in milliseconds")
    
    class Config:
        example = {
            "query": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that learns from data...",
            "sources": ["doc1_0", "doc1_1"],
            "has_sufficient_context": True,
            "num_chunks_retrieved": 2,
            "retrieval_metadata": [
                {"chunk_id": "doc1_0", "fusion_score": 0.95, "rank": 1}
            ],
            "latency_ms": 125.5
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    detail: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        example = {
            "error": "Invalid query",
            "status_code": 400,
            "detail": "Query cannot be empty"
        }
