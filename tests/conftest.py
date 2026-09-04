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
    from src.efficient_rag.milestone1 import DocumentProcessor
    return DocumentProcessor(chunk_size=500, chunk_overlap=50)
