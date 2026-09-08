"""Tests for M6 API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


def test_health_check(patch_rag_pipeline):
    """Test /health endpoint."""
    from efficient_rag.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_post_not_allowed(patch_rag_pipeline):
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


def test_query_response_structure(patch_rag_pipeline):
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


def test_query_with_results(patch_rag_pipeline):
    """Test query response with results."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": "test", "top_k": 2})
    data = response.json()
    
    assert response.status_code == 200
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) >= 0


def test_openapi_docs(patch_rag_pipeline):
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


def test_invalid_endpoint(patch_rag_pipeline):
    """Test nonexistent endpoint returns 404."""
    from efficient_rag.api.main import app
    client = TestClient(app)
    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_invalid_json(patch_rag_pipeline):
    """Test invalid JSON returns error."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", content="not json")
    assert response.status_code == 422


def test_missing_required_field(patch_rag_pipeline):
    """Test missing required field returns 422."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"top_k": 5})  # missing query
    assert response.status_code == 422


def test_latency_measured(patch_rag_pipeline):
    """Test that latency is measured."""
    from efficient_rag.api.main import app
    
    client = TestClient(app)
    response = client.post("/query", json={"query": "test", "top_k": 3})
    data = response.json()
    
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0


def test_metadata_in_response(patch_rag_pipeline):
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


def test_query_without_api_key_when_not_required(patch_rag_pipeline):
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


def test_query_missing_api_key_when_required(patch_rag_pipeline):
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


def test_query_invalid_api_key(patch_rag_pipeline):
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


def test_query_valid_api_key(patch_rag_pipeline):
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
