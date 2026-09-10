"""
M6 FastAPI application: Production API for RAG pipeline.
"""

import os
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
from .schemas import QueryRequest, QueryResponse, HealthResponse, ErrorResponse
from .dependencies import RAGPipeline, get_rag_pipeline

# Configuration
API_TITLE = "Efficient RAG System API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Production API for Retrieval-Augmented Generation with M1-M5 pipeline"

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Cache configuration
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
CACHE_MAX_SIZE = int(os.environ.get("CACHE_MAX_SIZE", "1000"))

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


class QueryCache:
    """Simple in-memory TTL cache for query responses."""
    
    def __init__(self, ttl_seconds: int, max_size: int):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        # Cache: {cache_key: (response_dict, timestamp)}
        self.cache = {}
    
    def _make_key(self, api_key: str, query: str, top_k: int) -> str:
        """Create cache key from API key, normalized query, and top_k."""
        # Normalize query: lowercase, strip whitespace, collapse multiple spaces
        normalized_query = " ".join(query.lower().split())
        # Don't include the actual API key in the cache key (only its hash)
        key_hash = hash(api_key) % (2**32)
        return f"{key_hash}:{normalized_query}:{top_k}"
    
    def get(self, api_key: str, query: str, top_k: int) -> Optional[dict]:
        """Get cached response if it exists and hasn't expired."""
        key = self._make_key(api_key, query, top_k)
        
        if key not in self.cache:
            return None
        
        response_dict, timestamp = self.cache[key]
        age = time.time() - timestamp
        
        # Check if expired
        if age > self.ttl_seconds:
            del self.cache[key]
            return None
        
        return response_dict
    
    def set(self, api_key: str, query: str, top_k: int, response_dict: dict) -> None:
        """Store response in cache with TTL."""
        # Enforce max size: remove oldest entry if at capacity
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = self._make_key(api_key, query, top_k)
        self.cache[key] = (response_dict, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global cache instance
query_cache = QueryCache(CACHE_TTL_SECONDS, CACHE_MAX_SIZE)


class RateLimiter:
    """Simple in-memory rate limiter per API key."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Track requests: {api_key: [(timestamp, ...)]}
        self.requests = defaultdict(list)
    
    def is_allowed(self, api_key: str) -> bool:
        """Check if request is allowed for this API key."""
        now = time.time()
        
        # Remove old requests outside the window
        self.requests[api_key] = [
            req_time for req_time in self.requests[api_key]
            if now - req_time < self.window_seconds
        ]
        
        # Check if we can allow this request
        if len(self.requests[api_key]) < self.max_requests:
            self.requests[api_key].append(now)
            return True
        
        return False
    
    def get_retry_after(self, api_key: str) -> int:
        """Get retry-after seconds for this API key."""
        if not self.requests[api_key]:
            return self.window_seconds
        
        oldest = self.requests[api_key][0]
        retry_after = int(self.window_seconds - (time.time() - oldest)) + 1
        return max(1, min(self.window_seconds, retry_after))


# Global rate limiter instance
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


class ResponseCache:
    """Simple in-memory TTL cache for /query responses, bounded by max size."""
    
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        # Cache: {cache_key: (response_dict, timestamp)}
        self.cache = {}
    
    def _make_key(self, api_key: str, query: str, top_k: int) -> str:
        """Create a cache key from API key, normalized query, and top_k."""
        # Normalize query: lowercase, strip whitespace
        normalized_query = query.strip().lower()
        return f"{api_key}:{normalized_query}:{top_k}"
    
    def get(self, api_key: str, query: str, top_k: int) -> Optional[dict]:
        """Get cached response if available and not expired."""
        key = self._make_key(api_key, query, top_k)
        
        if key not in self.cache:
            return None
        
        response, timestamp = self.cache[key]
        
        # Check if expired
        if time.time() - timestamp > self.ttl_seconds:
            # Remove expired entry
            del self.cache[key]
            return None
        
        return response
    
    def set(self, api_key: str, query: str, top_k: int, response: dict) -> None:
        """Cache a response with TTL."""
        key = self._make_key(api_key, query, top_k)
        
        # If cache is full, remove oldest entry (simple eviction)
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove oldest timestamp
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (response, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
    
    def size(self) -> int:
        """Return current number of cached entries."""
        return len(self.cache)


# Global cache instance
response_cache = ResponseCache(CACHE_MAX_SIZE, CACHE_TTL_SECONDS)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Verify X-API-Key header for protected endpoints.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        Verified API key
        
    Raises:
        HTTPException 401: If key missing or invalid
    """
    expected_key = os.environ.get("RAG_API_KEY")
    
    if not expected_key:
        # If RAG_API_KEY not configured, allow public access
        return "public"
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key


async def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """
    Check rate limit for API key on POST /query.
    
    Args:
        api_key: Verified API key
        
    Returns:
        API key if allowed
        
    Raises:
        HTTPException 429: If rate limit exceeded
    """
    if not rate_limiter.is_allowed(api_key):
        retry_after = rate_limiter.get_retry_after(api_key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds",
            headers={"Retry-After": str(retry_after)}
        )
    
    return api_key


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (public, no authentication or rate limiting)."""
    return HealthResponse(status="healthy", version=API_VERSION)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    api_key: str = Depends(check_rate_limit)
) -> QueryResponse:
    """
    Query the RAG system (requires X-API-Key header if RAG_API_KEY is configured).
    Subject to rate limiting: RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS.
    Responses are cached per (API key, query, top_k) with TTL.
    
    Args:
        request: Query request with query text and top_k
        pipeline: RAG pipeline (injected dependency)
        api_key: Validated API key from header (rate limit checked)
    
    Returns:
        QueryResponse with answer, sources, and metadata (cached or fresh)
    """
    try:
        # Check cache if enabled
        if CACHE_ENABLED:
            cached_response = query_cache.get(api_key, request.query, request.top_k)
            if cached_response is not None:
                # Return cached response
                return QueryResponse(**cached_response)
        
        # Cache miss or caching disabled: execute pipeline
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
        
        response_dict = {
            "query": result["query"],
            "answer": result["answer"],
            "sources": result["sources"],
            "has_sufficient_context": result["has_sufficient_context"],
            "num_chunks_retrieved": result["num_chunks_retrieved"],
            "retrieval_metadata": metadata,
            "latency_ms": latency_ms
        }
        
        # Cache successful response if enabled
        if CACHE_ENABLED:
            query_cache.set(api_key, request.query, request.top_k, response_dict)
        
        return QueryResponse(**response_dict)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query/stream")
async def query_stream_endpoint(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    api_key: str = Depends(check_rate_limit)
):
    """
    Stream query results with real LLM streaming.
    
    Streams generated text as Server-Sent Events (text/event-stream).
    Uses actual provider-level streaming, not simulated.
    
    Args:
        request: Query request with query text and top_k
        pipeline: RAG pipeline (injected dependency)
        api_key: Validated API key (rate limit checked)
    
    Returns:
        StreamingResponse with streamed text chunks and final metadata
    """
    async def stream_generator():
        try:
            # Perform RAG retrieval (not cached for streaming)
            context = pipeline.hybrid_retriever.search(request.query, top_k=request.top_k)
            
            # Build context string
            context_text = "\n\n".join([f"[{r['chunk_id']}] {r['text']}" for r in context])
            
            # Create grounded prompt
            prompt = f"""You are a helpful assistant. Answer the following question using ONLY the provided context.
If the context does not contain enough information to answer the question, say so explicitly.
Do not use any outside knowledge.

Context:
{context_text}

Question: {request.query}

Answer:"""
            
            # Stream the response
            llm = pipeline._rag_generator.llm
            full_response = ""
            
            for chunk in llm.stream(prompt, max_tokens=512):
                full_response += chunk
                # Send chunk as SSE
                yield f"data: {chunk}\n\n"
            
            # Send final metadata
            metadata = {
                "sources": list(set([r["chunk_id"] for r in context])),
                "num_chunks_retrieved": len(context),
                "has_sufficient_context": len(context_text) > 100
            }
            
            yield f"data: [DONE]\n\n"
            yield f"event: metadata\ndata: {str(metadata)}\n\n"
        
        except ValueError as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        except RuntimeError as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: Internal server error\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")



if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
