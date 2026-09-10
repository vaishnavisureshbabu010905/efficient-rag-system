"""Tests for M6 API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


def test_health_check(patch_rag_pipeline, reset_rate_limiter):
    """Test /health endpoint."""
    from efficient_rag.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_post_not_allowed(patch_rag_pipeline, reset_rate_limiter):
    """Test POST to /health returns 405."""
    from efficient_rag.api.main import app
    client = TestClient(app)
    response = client.post("/health", json={})
    assert response.status_code == 405


@pytest.mark.parametrize("top_k,should_fail", [
    (0, True),
    (1, False),
    (5, False),
    (50, False),
    (51, True),
])
def test_query_top_k_validation(patch_rag_pipeline, top_k, should_fail):
    """Test top_k parameter validation."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": "test", "top_k": top_k})
    
    if should_fail:
        assert response.status_code == 422
    else:
        assert response.status_code == 200


@pytest.mark.parametrize("query,should_fail", [
    ("", True),
    ("   ", True),
    ("test", False),
    ("What is machine learning?", False)
])
def test_query_validation(patch_rag_pipeline, query, should_fail):
    """Test query parameter validation."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": query, "top_k": 5})
    
    if should_fail:
        assert response.status_code == 422
    else:
        assert response.status_code == 200


def test_query_response_structure(patch_rag_pipeline, reset_rate_limiter):
    """Test query response has correct structure."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": "test", "top_k": 3})
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "has_sufficient_context" in data
    assert "num_chunks_retrieved" in data
    assert "latency_ms" in data


def test_query_with_results(patch_rag_pipeline, reset_rate_limiter):
    """Test query response with results."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": "test", "top_k": 2})
    data = response.json()
    
    assert response.status_code == 200
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) >= 0


def test_openapi_docs(patch_rag_pipeline, reset_rate_limiter):
    """Test OpenAPI documentation endpoints."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # Test /docs
    response = client.get("/docs")
    assert response.status_code == 200
    
    # Test /redoc
    response = client.get("/redoc")
    assert response.status_code == 200
    
    # Test /openapi.json
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema or "paths" in schema


def test_invalid_endpoint(patch_rag_pipeline, reset_rate_limiter):
    """Test nonexistent endpoint returns 404."""
    from efficient_rag.api.main import app
    client = TestClient(app)
    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_invalid_json(patch_rag_pipeline, reset_rate_limiter):
    """Test invalid JSON returns error."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", content="not json")
    assert response.status_code == 422


def test_missing_required_field(patch_rag_pipeline, reset_rate_limiter):
    """Test missing required field returns 422."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"top_k": 5})  # missing query
    assert response.status_code == 422


def test_latency_measured(patch_rag_pipeline, reset_rate_limiter):
    """Test that latency is measured."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": "test", "top_k": 3})
    data = response.json()
    
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0


def test_metadata_in_response(patch_rag_pipeline, reset_rate_limiter):
    """Test retrieval metadata in response."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": "test", "top_k": 2})
    data = response.json()
    
    assert "retrieval_metadata" in data
    if data["num_chunks_retrieved"] > 0:
        assert len(data["retrieval_metadata"]) > 0


def test_api_uses_correct_imports():
    """Regression test: verify FastAPI dependencies import from correct package path."""
    # This test ensures the import fix (src.efficient_rag -> efficient_rag) is in place
    # If the fix is missing, this will fail
    import importlib.util
    import os
    
    # Check that dependencies.py uses the correct import path
    spec = importlib.util.spec_from_file_location("dependencies", 
                                                    "src/efficient_rag/api/dependencies.py")
    dependencies_module = importlib.util.module_from_spec(spec)
    
    # Read the file and check imports
    with open("src/efficient_rag/api/dependencies.py", "r") as f:
        content = f.read()
    
    # Should use efficient_rag, not src.efficient_rag
    assert "from efficient_rag." in content, "dependencies.py should import from efficient_rag"
    assert "from src.efficient_rag" not in content, "dependencies.py should not import from src.efficient_rag"


def test_query_without_api_key_when_not_required(patch_rag_pipeline, reset_rate_limiter):
    """Test /query works without API key when RAG_API_KEY not configured."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # RAG_API_KEY not set - authentication not required
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAG_API_KEY", None)
        response = client.post("/query", json={"query": "test", "top_k": 5})
        assert response.status_code == 200


def test_query_missing_api_key_when_required(patch_rag_pipeline, reset_rate_limiter):
    """Test /query returns 401 when API key required but missing."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # RAG_API_KEY is set - authentication required
    with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}, clear=False):
        # No X-API-Key header
        response = client.post("/query", json={"query": "test", "top_k": 5})
        assert response.status_code == 401
        assert "Missing X-API-Key header" in response.json()["detail"]


def test_query_invalid_api_key(patch_rag_pipeline, reset_rate_limiter):
    """Test /query returns 401 when API key is invalid."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # RAG_API_KEY is set - authentication required
    with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}, clear=False):
        # Wrong X-API-Key header
        response = client.post(
            "/query",
            json={"query": "test", "top_k": 5},
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


def test_query_valid_api_key(patch_rag_pipeline, reset_rate_limiter):
    """Test /query succeeds with correct API key."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # RAG_API_KEY is set - authentication required
    with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}, clear=False):
        # Correct X-API-Key header
        response = client.post(
            "/query",
            json={"query": "test", "top_k": 5},
            headers={"X-API-Key": "secret-key"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data


def test_health_always_public():
    """Test /health is always public (no authentication required)."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # Even with RAG_API_KEY set, /health should be accessible
    with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}, clear=False):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_docs_always_public():
    """Test /docs is always public (no authentication required)."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # Even with RAG_API_KEY set, /docs should be accessible
    with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}, clear=False):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()


def test_rate_limiter_tracks_requests():
    """Test RateLimiter correctly tracks requests per API key."""
    from efficient_rag.api.main import RateLimiter
    
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    
    # First two requests should be allowed
    assert limiter.is_allowed("api_key_1") is True
    assert limiter.is_allowed("api_key_1") is True
    
    # Third request should be denied
    assert limiter.is_allowed("api_key_1") is False
    
    # Different key should have independent limit
    assert limiter.is_allowed("api_key_2") is True


def test_rate_limiter_independent_keys():
    """Test different API keys have independent rate limits."""
    from efficient_rag.api.main import RateLimiter
    
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    
    # key1: 1 request allowed, 2nd denied
    assert limiter.is_allowed("key1") is True
    assert limiter.is_allowed("key1") is False
    
    # key2: should have its own limit
    assert limiter.is_allowed("key2") is True
    assert limiter.is_allowed("key2") is False


def test_rate_limiter_retry_after():
    """Test RateLimiter provides correct Retry-After value."""
    from efficient_rag.api.main import RateLimiter
    import time
    
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    
    # Use up the limit
    assert limiter.is_allowed("key1") is True
    
    # Get retry-after value
    retry_after = limiter.get_retry_after("key1")
    
    # Should be between 1 and 10 seconds
    assert 1 <= retry_after <= 10


def test_health_endpoint_not_rate_limited():
    """Test /health endpoint is never rate limited."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    # Make multiple requests to /health
    # They should all succeed regardless of rate limiting
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code == 200


def test_query_cache_structure():
    """Test ResponseCache basic structure and operations."""
    from efficient_rag.api.main import QueryCache
    
    cache = QueryCache(max_size=2, ttl_seconds=10)
    
    # Initially empty
    assert cache.size() == 0
    
    # Set and get
    response = {"answer": "test", "sources": []}
    cache.set("key1", "query", 5, response)
    
    assert cache.size() == 1
    cached = cache.get("key1", "query", 5)
    assert cached == response


def test_query_cache_miss():
    """Test ResponseCache returns None on miss."""
    from efficient_rag.api.main import QueryCache
    
    cache = QueryCache(max_size=10, ttl_seconds=10)
    
    result = cache.get("key1", "query", 5)
    assert result is None


def test_query_cache_ttl_expiry():
    """Test ResponseCache entries expire after TTL."""
    from efficient_rag.api.main import QueryCache
    import time
    
    cache = QueryCache(max_size=10, ttl_seconds=1)
    
    response = {"answer": "test", "sources": []}
    cache.set("key1", "query", 5, response)
    
    # Immediately, should be cached
    assert cache.get("key1", "query", 5) is not None
    
    # Wait for expiry
    time.sleep(1.1)
    
    # Should now be expired
    assert cache.get("key1", "query", 5) is None


def test_query_cache_different_keys():
    """Test different cache keys are independent."""
    from efficient_rag.api.main import QueryCache
    
    cache = QueryCache(max_size=10, ttl_seconds=10)
    
    response1 = {"answer": "test1", "sources": []}
    response2 = {"answer": "test2", "sources": []}
    
    cache.set("key1", "query", 5, response1)
    cache.set("key2", "query", 5, response2)
    
    # Different API key
    assert cache.get("key1", "query", 5) == response1
    assert cache.get("key2", "query", 5) == response2
    
    # Different top_k
    assert cache.get("key1", "query", 10) is None
    
    # Different query
    assert cache.get("key1", "other", 5) is None


def test_query_cache_query_normalization():
    """Test cache key normalizes queries (lowercase, whitespace)."""
    from efficient_rag.api.main import QueryCache
    
    cache = QueryCache(max_size=10, ttl_seconds=10)
    
    response = {"answer": "test", "sources": []}
    
    # Set with one variant
    cache.set("key1", "Hello World", 5, response)
    
    # Get with different case/whitespace should hit
    assert cache.get("key1", "hello world", 5) == response
    assert cache.get("key1", "HELLO  WORLD", 5) == response
    assert cache.get("key1", "  hello world  ", 5) == response


def test_query_cache_max_size_eviction():
    """Test cache evicts oldest on max size."""
    from efficient_rag.api.main import QueryCache
    
    cache = QueryCache(max_size=2, ttl_seconds=10)
    
    # Fill cache
    cache.set("key1", "q1", 5, {"answer": "1"})
    cache.set("key1", "q2", 5, {"answer": "2"})
    
    assert cache.size() == 2
    
    # Add third, should evict oldest
    cache.set("key1", "q3", 5, {"answer": "3"})
    
    assert cache.size() == 2
    # q1 should be evicted
    assert cache.get("key1", "q1", 5) is None




def test_stream_endpoint_authenticated(patch_rag_pipeline, reset_rate_limiter, reset_query_cache):
    """Test streaming endpoint with valid API key."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}, clear=False):
        response = client.post(
            "/query/stream",
            json={"query": "test", "top_k": 5},
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


def test_stream_endpoint_missing_api_key(patch_rag_pipeline):
    """Test streaming endpoint returns 401 without API key."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}, clear=False):
        response = client.post(
            "/query/stream",
            json={"query": "test", "top_k": 5}
        )
        assert response.status_code == 401


def test_stream_endpoint_invalid_api_key(patch_rag_pipeline):
    """Test streaming endpoint returns 401 with invalid API key."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    with patch.dict(os.environ, {"RAG_API_KEY": "correct-key"}, clear=False):
        response = client.post(
            "/query/stream",
            json={"query": "test", "top_k": 5},
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 401


def test_stream_endpoint_returns_event_stream(patch_rag_pipeline, reset_rate_limiter, reset_query_cache):
    """Test streaming endpoint returns Server-Sent Events format."""
    import os
    from unittest.mock import patch
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}, clear=False):
        response = client.post(
            "/query/stream",
            json={"query": "test", "top_k": 5},
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        content = response.text
        # Should contain data: events
        assert "data:" in content


def test_existing_query_endpoint_unchanged(patch_rag_pipeline, reset_rate_limiter, reset_query_cache):
    """Test non-streaming /query endpoint still works."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    
    response = client.post("/query", json={"query": "test", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
