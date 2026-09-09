"""Pytest configuration and fixtures for M1 tests"""

import pytest
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


@pytest.fixture
def sample_documents_dir(tmp_path):
    """Create temporary directory with sample PDF and TXT files"""
    
    # Create sample TXT
    txt_file = tmp_path / "sample.txt"
    txt_content = "This is a sample text document. " * 30
    txt_file.write_text(txt_content, encoding="utf-8")
    
    # Create sample PDF with 2 pages
    pdf_file = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_file), pagesize=letter)
    
    # Page 1
    c.drawString(100, 750, "Sample PDF Document - Page 1")
    c.drawString(100, 730, "This is test content on page 1. " * 20)
    c.showPage()
    
    # Page 2
    c.drawString(100, 750, "Sample PDF Document - Page 2")
    c.drawString(100, 730, "This is test content on page 2. " * 20)
    c.showPage()
    
    c.save()
    
    return tmp_path


@pytest.fixture
def document_processor():
    """Create DocumentProcessor instance for testing"""
    from efficient_rag.milestone1 import DocumentProcessor
    return DocumentProcessor(chunk_size=500, chunk_overlap=50)


# M6 API fixtures
@pytest.fixture
def mock_rag_pipeline_for_api():
    """Mock RAG pipeline for API tests (not auto-applied)."""
    from unittest.mock import Mock
    
    mock_pipeline = Mock()
    mock_pipeline.query = Mock(return_value={
        "query": "test query",
        "answer": "test answer",
        "sources": ["c0"],
        "has_sufficient_context": True,
        "num_chunks_retrieved": 1,
        "retrieval_results": [{"chunk_id": "c0", "fusion_score": 0.9, "rank": 1}]
    })
    return mock_pipeline




@pytest.fixture
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    from efficient_rag.api.main import rate_limiter
    rate_limiter.requests.clear()
    yield
    rate_limiter.requests.clear()


@pytest.fixture
def reset_query_cache():
    """Reset query cache before each test."""
    from efficient_rag.api.main import query_cache
    query_cache.clear()
    yield
    query_cache.clear()


@pytest.fixture
def patch_rag_pipeline(mock_rag_pipeline_for_api):
    """Patch RAG pipeline for API tests."""
    from unittest.mock import patch
    
    with patch("efficient_rag.api.dependencies.RAGPipeline") as mock_class:
        mock_class.return_value = mock_rag_pipeline_for_api
        with patch("efficient_rag.api.dependencies.get_rag_pipeline", return_value=mock_rag_pipeline_for_api):
            yield mock_rag_pipeline_for_api
