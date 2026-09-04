# Efficient RAG System with Binary Quantization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A production-quality Retrieval Augmented Generation (RAG) system with binary quantization, built incrementally with clean architecture and comprehensive testing.

## 🎯 Project Overview

This project implements an efficient RAG pipeline optimized for:
- **Scalability**: Binary quantization for fast approximate retrieval
- **Efficiency**: Optimized chunking and indexing
- **Quality**: Comprehensive testing and documentation
- **Modularity**: Incremental milestone-based development

## 📦 Current Status: Milestone 1 ✅

**Milestone 1: PDF/TXT Ingestion & Character-Based Chunking**

Implements the foundational layer of the RAG pipeline:
```
Documents (PDF/TXT) → Text Extraction → Chunking → Metadata → JSON Output
```

### M1 Features

✅ **Multi-Format Ingestion**
- PDF files with pdfplumber
- TXT files with UTF-8 encoding
- Automatic file discovery
- Graceful error handling

✅ **Smart Text Extraction**
- PDF page boundary preservation
- Clean text extraction
- Encoding fallback handling

✅ **Configurable Chunking**
- Character-based chunking
- Adjustable chunk size & overlap
- Overlap validation (overlap < size)

✅ **Rich Metadata**
- Filename, file type, size
- Modification timestamp
- Page count (PDFs)
- Character position tracking

✅ **Stable Chunk IDs**
- Format: `{filename}_{chunk_number}`
- Document-specific
- Deterministic & reproducible

✅ **Production Quality**
- Type hints & docstrings
- Error handling & validation
- 16 unit tests (82% coverage)
- Comprehensive documentation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/vaishnavisureshbabu010905/efficient-rag-system.git
cd efficient-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Development mode
```

### Run Example

```bash
# Process sample documents
python -c "
from src.efficient_rag.milestone1.document_processor import DocumentProcessor
from pathlib import Path

processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
chunks = processor.process_pipeline('data/sample_documents')
print(f'Created {len(chunks)} chunks')
"
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/efficient_rag --cov-report=term-missing

# Run specific test
pytest tests/test_milestone1.py::TestDocumentProcessor::test_ingest_documents -v
```

## 📁 Project Structure

```
efficient-rag-system/
├── README.md                                    # This file
├── requirements.txt                             # Dependencies
├── setup.py                                     # Package config
├── .gitignore                                   # Git rules
│
├── src/efficient_rag/
│   ├── __init__.py
│   └── milestone1/
│       ├── __init__.py
│       └── document_processor.py               # M1: Core logic (230 lines)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                             # Pytest fixtures
│   └── test_milestone1.py                       # 16 unit tests
│
├── docs/
│   └── MILESTONE_1.md                          # Detailed M1 documentation
│
├── data/
│   ├── sample_documents/                       # Sample input files
│   │   ├── sample.txt
│   │   └── sample.pdf
│   └── outputs/                                # Generated output
│       └── .gitkeep
```

## 💻 Usage

### Basic Usage

```python
from src.efficient_rag.milestone1.document_processor import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Character overlap
)

# Process documents
chunks = processor.process_pipeline("path/to/documents")

# Access results
for chunk in chunks:
    print(f"ID: {chunk['metadata']['chunk_id']}")
    print(f"Text: {chunk['text'][:100]}...")
```

### Output Format

Each chunk is a dictionary with:
```python
{
    "text": "Chunk content...",
    "metadata": {
        "filename": "document.txt",
        "file_type": ".txt",
        "file_size_bytes": 3842,
        "modified_date": "2024-01-15T10:30:45.123456",
        "chunk_id": "document.txt_0",
        "char_start": 0,
        "char_end": 500
    }
}
```

## 🧪 Testing

### Test Coverage

- **16 unit tests** covering:
  - Initialization & validation
  - PDF/TXT ingestion
  - Text extraction
  - Chunking logic
  - Metadata extraction
  - Stable ID generation
  - Edge cases (empty folders, small chunks, etc.)

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/efficient_rag --cov-report=html
open htmlcov/index.html  # View HTML report

# Specific test class
pytest tests/test_milestone1.py::TestDocumentProcessor -v

# Specific test
pytest tests/test_milestone1.py::TestDocumentProcessor::test_ingest_documents -v
```

## 📊 Configuration

### Chunk Size & Overlap

```python
# Fine-grained chunking (more, smaller chunks)
processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

# Coarse-grained chunking (fewer, larger chunks)
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)

# No overlap (minimal redundancy)
processor = DocumentProcessor(chunk_size=500, chunk_overlap=0)

# Default (recommended)
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
```

## 🔧 Development

### Add Your Own Documents

```bash
# Copy documents to sample directory
cp /path/to/document.pdf data/sample_documents/
cp /path/to/document.txt data/sample_documents/

# Process them
python -c "
from src.efficient_rag.milestone1.document_processor import DocumentProcessor
processor = DocumentProcessor()
chunks = processor.process_pipeline('data/sample_documents')
print(f'Created {len(chunks)} chunks')
"
```

### Install in Development Mode

```bash
pip install -e .
pip install -r requirements.txt
```

### Run Full Test Suite

```bash
pytest tests/ -v --cov=src/efficient_rag --cov-report=term-missing
```

## 📚 Documentation

- **README.md** (this file) - Quick start & overview
- **docs/MILESTONE_1.md** - Detailed M1 documentation
- **Code docstrings** - Inline function documentation
- **tests/test_milestone1.py** - Test cases as usage examples

## 🎯 Milestones

### ✅ Milestone 1: Document Ingestion & Chunking
- PDF/TXT ingestion
- Character-based chunking
- Metadata extraction
- JSON output

### 📋 Milestone 2: Embeddings & Vector Database
- LLM integration (OpenAI, HuggingFace)
- Vector database setup (FAISS, Pinecone)
- Semantic search

### 📋 Milestone 3: Binary Quantization
- Vector quantization
- Hamming distance search
- Compression analysis

### 📋 Milestone 4: Retrieval & Ranking
- Hybrid BM25 + semantic search
- Re-ranking strategies

### 📋 Milestone 5: RAG Integration
- LLM integration for generation
- Prompt engineering

### 📋 Milestone 6: Production Deployment
- FastAPI service
- Docker containerization
- Cloud deployment

## 🤝 Contributing

This is a portfolio project. Contributions welcome for:
- Additional file format support (DOCX, PPTX)
- Performance optimizations
- Test coverage improvements
- Documentation enhancements

## 📄 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

Built as an AI Engineer portfolio project demonstrating:
- Python development best practices
- Software architecture & design
- Testing & quality assurance
- Incremental development methodology

## 📞 Support

For issues or questions:
1. Check `docs/MILESTONE_1.md` for detailed information
2. Review test cases in `tests/test_milestone1.py`
3. Check inline code documentation

---

**Status:** Milestone 1 Complete ✅ | Ready for M2 Planning 🚀

**Last Updated:** September 4, 2024  
**Python Version:** 3.8+  
**Test Status:** 16/16 PASSED
