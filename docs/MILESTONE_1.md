# Milestone 1: PDF/TXT Ingestion & Character-Based Chunking

## Overview

Milestone 1 implements the foundational layer of the RAG pipeline:

```
Documents (PDF/TXT) → Text Extraction → Chunking → Metadata → JSON Output
```

## Features

### ✅ Multi-Format Ingestion

**PDF Files**
- Multi-page PDF support using pdfplumber
- Page-by-page text extraction
- Page boundary preservation (pages joined with `\n\n`)
- Page count metadata tracking
- Error handling for corrupted PDFs

**TXT Files**
- UTF-8 encoding support
- Fallback encoding handling
- Direct file reading
- Encoding error handling

**File Discovery**
- Automatic scanning of folder
- Recursive file filtering (PDF, TXT only)
- Case-insensitive extension matching
- Sorted file processing

### ✅ Smart Text Extraction

**PDF Text Extraction**
```python
# Pages are joined with double newlines to preserve boundaries
text = "\n\n".join(page.extract_text() for page in pdf.pages)
```

Benefits:
- Maintains document structure
- Prevents mid-sentence merging
- Preserves paragraph boundaries
- Supports text-based chunking

**TXT Text Extraction**
```python
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
```

**Error Handling**
- Graceful degradation on errors
- Continues processing other files
- Prints warnings for failed files
- Returns empty list on error

### ✅ Character-Based Chunking

**Algorithm**
```python
# Sliding window with overlap
for i in range(0, len(text), chunk_size - chunk_overlap):
    chunk_text = text[i:i + chunk_size]
```

**Configuration**
- `chunk_size`: Characters per chunk (default: 500)
- `chunk_overlap`: Character overlap (default: 50)
- Validation: `overlap < size` enforced

**Benefits**
- Language-agnostic (works for any text)
- Deterministic (same input → same output)
- Fast (no external tokenizers needed)
- Simple to understand and modify

**Limitations**
- Token count varies by language model
- May break mid-word (addressed in M2 with token-based chunking)

### ✅ Metadata Extraction

**File Metadata**
- `filename`: Original file name with extension
- `file_type`: Extension (e.g., `.txt`, `.pdf`)
- `file_size_bytes`: File size in bytes
- `modified_date`: ISO format timestamp
- `pages`: Page count (PDFs only)

**Chunk Metadata**
- `chunk_id`: Stable identifier (`{filename}_{chunk_number}`)
- `char_start`: Starting character position in original text
- `char_end`: Ending character position in original text

**Example**
```json
{
  "filename": "document.txt",
  "file_type": ".txt",
  "file_size_bytes": 3842,
  "modified_date": "2024-01-15T10:30:45.123456",
  "chunk_id": "document.txt_0",
  "char_start": 0,
  "char_end": 500
}
```

### ✅ Stable Chunk IDs

**ID Format**
```
{filename}_{chunk_number}
```

**Examples**
- `document.txt_0` - First chunk of document.txt
- `document.txt_1` - Second chunk of document.txt
- `research.pdf_0` - First chunk of research.pdf
- `research.pdf_1` - Second chunk of research.pdf

**Properties**
- ✅ Deterministic (same input → same IDs)
- ✅ Document-specific (filename included)
- ✅ Globally unique across corpus
- ✅ Sequential (easy to order)
- ✅ Reproducible (consistent across runs)

**Use Cases**
- Reference individual chunks
- Track chunk provenance
- Enable chunk-level operations
- Support chunk-level caching

## Configuration

### Basic Configuration

```python
from src.efficient_rag.milestone1 import DocumentProcessor

# Default configuration
processor = DocumentProcessor(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Character overlap
)
```

### Parameter Recommendations

**Fine-Grained Chunking**
```python
processor = DocumentProcessor(
    chunk_size=200,      # Smaller chunks
    chunk_overlap=20     # 10% overlap
)
# Use for: Dense information, frequent chunk access
```

**Coarse-Grained Chunking**
```python
processor = DocumentProcessor(
    chunk_size=1000,     # Larger chunks
    chunk_overlap=100    # 10% overlap
)
# Use for: Long documents, infrequent chunk access
```

**No Overlap**
```python
processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=0      # No overlap
)
# Use for: Memory-constrained environments, minimal redundancy
```

**High Overlap**
```python
processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=250    # 50% overlap
)
# Use for: Maximum context preservation, higher redundancy
```

## Usage Examples

### Example 1: Basic Usage

```python
from src.efficient_rag.milestone1 import DocumentProcessor

# Initialize
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

# Process documents
chunks = processor.process_pipeline("path/to/documents")

# Access results
for chunk in chunks:
    print(f"ID: {chunk['metadata']['chunk_id']}")
    print(f"Text: {chunk['text'][:100]}...")
    print(f"Position: {chunk['metadata']['char_start']}-{chunk['metadata']['char_end']}")
```

### Example 2: Custom Configuration

```python
from src.efficient_rag.milestone1 import DocumentProcessor

# Fine-grained chunking
processor = DocumentProcessor(chunk_size=256, chunk_overlap=25)

# Process
chunks = processor.process_pipeline("data/sample_documents", verbose=True)

print(f"Created {len(chunks)} chunks")
```

### Example 3: Advanced - Programmatic Pipeline

```python
from src.efficient_rag.milestone1 import DocumentProcessor

processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

# Step 1: Ingest
documents = processor.ingest_documents("path/to/documents")
print(f"Ingested {len(documents)} documents")

# Step 2: Chunk
chunks = processor.chunk_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Save to JSON
import json
with open("chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)
```

## Output Format

### JSON Structure

```json
[
  {
    "text": "Full chunk text content here...",
    "metadata": {
      "filename": "document.txt",
      "file_type": ".txt",
      "file_size_bytes": 3842,
      "modified_date": "2024-01-15T10:30:45.123456",
      "chunk_id": "document.txt_0",
      "char_start": 0,
      "char_end": 500
    }
  },
  {
    "text": "Next chunk content here...",
    "metadata": {
      "filename": "document.txt",
      "file_type": ".txt",
      "file_size_bytes": 3842,
      "modified_date": "2024-01-15T10:30:45.123456",
      "chunk_id": "document.txt_1",
      "char_start": 450,
      "char_end": 950
    }
  }
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The chunk content (up to chunk_size characters) |
| `filename` | string | Original file name with extension |
| `file_type` | string | File extension (`.txt`, `.pdf`) |
| `file_size_bytes` | int | File size in bytes |
| `modified_date` | string | ISO format timestamp (YYYY-MM-DDTHH:MM:SS) |
| `pages` | int | Page count (PDFs only) |
| `chunk_id` | string | Stable identifier (`{filename}_{number}`) |
| `char_start` | int | Starting character position in original text |
| `char_end` | int | Ending character position in original text |

## Testing

### Test Coverage

16 unit tests covering:

**Core Functionality (12 tests)**
- ✅ Initialization & validation
- ✅ PDF ingestion & extraction
- ✅ TXT ingestion & extraction
- ✅ Metadata extraction
- ✅ Character-based chunking
- ✅ Stable chunk ID generation
- ✅ Character position tracking
- ✅ Pipeline integration
- ✅ Chunk overlap validation
- ✅ Error handling (missing folder)

**Edge Cases (4 tests)**
- ✅ Empty folders
- ✅ Very small chunks
- ✅ Large overlap ratios
- ✅ Mixed file types (filtering)

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/efficient_rag --cov-report=term-missing

# Specific test class
pytest tests/test_milestone1.py::TestDocumentProcessor -v

# Specific test
pytest tests/test_milestone1.py::TestDocumentProcessor::test_ingest_documents -v
```

### Test Fixtures

**sample_documents_dir**
- Temporary directory with sample PDF and TXT files
- Auto-cleanup after test
- Used by most M1 tests

**document_processor**
- Pre-configured DocumentProcessor instance
- chunk_size=500, chunk_overlap=50
- Used for testing

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Ingestion | O(n) | n = number of files |
| Text extraction | O(m) | m = file size in bytes |
| Chunking | O(t) | t = total text length |
| Overall pipeline | O(n + m + t) | Linear in total data size |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Documents | O(m) | m = total file size |
| Chunks | O(m) | Same size as original text |
| Metadata | O(n * k) | n = chunks, k = metadata fields |
| Overall | O(m) | Linear in total data size |

### Benchmarks (Sample Data)

- **Ingestion**: ~10-50 MB/s (depends on file format)
- **Chunking**: ~1-10 MB/s (single-threaded)
- **Overall**: ~5-20 MB/s (combined pipeline)

## Design Decisions

### 1. Character-Based Chunking (Not Token-Based)

**Why**
- Language-agnostic (works for any language/script)
- Deterministic (no randomness)
- No external dependencies
- Simple and fast

**When to Change**
- M2: Token-based chunking after embeddings are available
- M2: Semantic chunking using embeddings

### 2. PDF Page Boundary Preservation

**Implementation**
```python
text = "\n\n".join(page.extract_text() for page in pdf.pages)
```

**Why**
- Maintains document structure
- Prevents sentences from merging across pages
- Preserves paragraph boundaries
- Enables document reconstruction

### 3. Stable Chunk IDs

**Format**: `{filename}_{chunk_number}`

**Why**
- Deterministic (consistent across runs)
- Document-specific (includes filename)
- Human-readable
- Enables reliable chunk tracking

### 4. Overlap Strategy

**Default**: 50 characters out of 500 (10% overlap)

**Why**
- Preserves context across chunk boundaries
- Minimal redundancy overhead
- Improves retrieval relevance
- Balanced default for most use cases

## Future Enhancements (M2+)

### M2: Embeddings & Vector Database
- LLM integration for embeddings
- Vector database indexing
- Semantic similarity search

### M3: Binary Quantization
- Vector quantization to binary
- Hamming distance search
- Storage optimization

### M4: Retrieval & Ranking
- Hybrid BM25 + semantic search
- Re-ranking strategies
- Result aggregation

### M5: RAG Integration
- LLM integration for generation
- Prompt engineering
- Q&A pipeline

### M6: Production Deployment
- FastAPI service
- Docker containerization
- Cloud deployment

## Troubleshooting

### PDF Extraction Issues

**Problem**: PDF extraction produces empty or garbled text

**Solutions**
1. Check PDF format (text-based vs. scanned image)
2. Verify pdfplumber installation: `pip install pdfplumber`
3. Test with different PDF file
4. Check file encoding/corruption

### TXT Encoding Issues

**Problem**: TXT file produces encoding errors

**Solutions**
1. Verify UTF-8 encoding: `file document.txt`
2. Convert to UTF-8: `iconv -f ISO-8859-1 -t UTF-8 file.txt`
3. Check for BOM markers: `hexdump -C file.txt | head`

### Memory Issues

**Problem**: Large files cause memory errors

**Solutions**
1. Increase chunk_size for fewer chunks
2. Process files sequentially (current behavior)
3. Split large files before processing

## Contributing

To contribute to M1:

1. Add new file format support (DOCX, PPTX)
2. Performance optimizations
3. Test coverage improvements
4. Documentation enhancements

## References

- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
- [Python pathlib](https://docs.python.org/3/library/pathlib.html)
- [Character encoding in Python](https://docs.python.org/3/library/codecs.html)

---

**Status**: Milestone 1 Complete ✅

**Last Updated**: September 4, 2024

**Next Milestone**: M2 - Embeddings & Vector Database
