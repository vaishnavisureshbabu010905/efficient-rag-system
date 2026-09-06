"""
M6 FastAPI application: Production API for RAG pipeline.
"""

import os
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from .schemas import QueryRequest, QueryResponse, HealthResponse, ErrorResponse
from .dependencies import RAGPipeline, get_rag_pipeline

# Configuration
API_TITLE = "Efficient RAG System API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Production API for Retrieval-Augmented Generation with M1-M5 pipeline"

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=API_VERSION)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> QueryResponse:
    """
    Query the RAG system.
    
    Args:
        request: Query request with query text and top_k
        pipeline: RAG pipeline (injected dependency)
    
    Returns:
        QueryResponse with answer, sources, and metadata
    """
    try:
        # Time the query
        start_time = time.time()
        
        # Execute query
        result = pipeline.query(request.query, top_k=request.top_k)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Build response
        metadata = [
            {
                "chunk_id": r.get("chunk_id"),
                "fusion_score": r.get("fusion_score"),
                "rank": r.get("rank")
            }
            for r in result.get("retrieval_results", [])
        ]
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            has_sufficient_context=result["has_sufficient_context"],
            num_chunks_retrieved=result["num_chunks_retrieved"],
            retrieval_metadata=metadata,
            latency_ms=latency_ms
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")





if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
