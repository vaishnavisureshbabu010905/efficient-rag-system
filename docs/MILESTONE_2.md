# Milestone 2: Dense Embedding + FAISS Retrieval

## Overview

M2 implements dense embedding-based retrieval using FAISS. It provides a semantic search baseline that will later be compared against:
- **M3**: Binary-quantized retrieval (approximate, fast)
- **M4**: Hybrid retrieval (dense + sparse BM25)

## Architecture

```
M1 Chunks (text + metadata)
           ↓
    DenseEmbedder
    (SentenceTransformer)
           ↓
    Dense Embeddings (384-dim)
           ↓
    DenseRetriever (FAISS)
           ↓
    Indexed + Searchable
           ↓
    Semantic Search (top-k)
```

## Components

### DenseEmbedder

Encodes text to dense vectors using SentenceTransformer.

```python
from src.efficient_rag.milestone2.embedding import DenseEmbedder

embedder = DenseEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Encode chunks
texts = ["text1", "text2", "text3"]
embeddings = embedder.encode_chunks(texts)  # (3, 384)

# Encode query
query_embedding = embedder.encode_query("search term")  # (384,)

print(f"Embedding dimension: {embedder.dim}")  # 384
```

**Model**: `all-MiniLM-L6-v2`
- Dimension: 384
- Embedding type: dense, L2-normalized
- Training: Sentence similarity using contrastive learning

### DenseRetriever

Indexes embeddings in FAISS and performs semantic search.

```python
from src.efficient_rag.milestone2.retriever import DenseRetriever

retriever = DenseRetriever(embedder)

# Build index
num_chunks = retriever.build_index(chunks)

# Search
results = retriever.search("query text", top_k=5)
# Returns: [{text, metadata, chunk_id, score}, ...]

# Persistence
retriever.save_index('path/to/index')
retriever.load_index('path/to/index')
```

**Index Type**: FlatIP (IndexFlatIP)
- Exact inner product search
- Works with L2-normalized embeddings for cosine similarity
- No quantization or approximation

**Search Flow**:
1. Encode query with DenseEmbedder
2. Query FAISS index (inner product)
3. Return top-k with scores (similarity: 0.0 to 1.0)

## Integration with M1

M1 output (chunks with metadata) → M2 input:

```python
from src.efficient_rag.milestone1.document_processor import DocumentProcessor
from src.efficient_rag.milestone2.embedding import DenseEmbedder
from src.efficient_rag.milestone2.retriever import DenseRetriever

# M1: Process documents
processor = DocumentProcessor()
chunks = processor.process_pipeline('data/sample_documents')

# M2: Build dense index
embedder = DenseEmbedder()
retriever = DenseRetriever(embedder)
num_indexed = retriever.build_index(chunks)

# M2: Search
results = retriever.search("machine learning", top_k=5)
```

## Persistence

### Save

```python
retriever.save_index('data/outputs/m2_index')
```

Saves:
- `index.faiss`: FAISS index binary
- `metadata.json`: Chunk IDs, texts, metadata

### Load

```python
retriever.load_index('data/outputs/m2_index')
results = retriever.search("query")  # Works immediately
```

## Benchmarks (Measured)

Sample documents: 13 chunks

| Metric | Value | Notes |
|--------|-------|-------|
| Chunks indexed | 13 | From M1 output |
| Index build time | 0.18s | Embedding + FAISS indexing |
| Query latency (avg) | 12.5ms | Top-5 search |
| Embedding dim | 384 | all-MiniLM-L6-v2 |
| Index type | FlatIP | Exact search |

**Scalability**: Linear with chunk count. For 1000 chunks, expect ~1.5-2s build, ~12-15ms query.

## Testing

14 tests covering:
- Embedding generation (shape, dtype, normalization)
- Index building (size, structure)
- Search (top-k, ordering, metadata preservation)
- Persistence (save/load roundtrip)
- Edge cases (empty chunks, invalid queries)

Status: **14/14 passing**

## Design Decisions

### L2-Normalized Embeddings + Inner Product = Cosine Similarity

Why normalize before indexing?
- Inner product of normalized vectors = cosine similarity
- FAISS FlatIP is optimized for this
- Simplifies post-processing (no additional normalization needed)

### No Approximate Methods in M2

Why exact search in M2?
- M2 is the baseline for comparison
- M3 will add binary quantization (approximate)
- Provides ground truth for evaluating M3

### Independent Embedder/Retriever

Why separate classes?
- DenseEmbedder can be swapped for other models
- DenseRetriever can use different indices (e.g., HNSW, IVF)
- Enables M3 and M4 to reuse both components

## Future Work (M3+)

**M3: Binary Quantization**
- Same embeddings as M2
- Quantize to binary (1-bit per dim)
- Use Hamming distance in FAISS
- Compare speed/accuracy vs M2

**M4: Hybrid Retrieval**
- Keep dense retrieval from M2
- Add sparse BM25 retrieval
- Combine rankings (dense + sparse)
- Evaluate hybrid quality

## References

- [SentenceTransformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Cosine Similarity & Normalization](https://en.wikipedia.org/wiki/Cosine_similarity)

---

**Status**: M2 Complete ✅  
**Tests**: 14/14 passing  
**Ready for M3**: Yes
