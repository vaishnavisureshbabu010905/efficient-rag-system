# Milestone 3: Binary Quantization + Hamming Retrieval

## Overview

M3 implements 1-bit quantization of dense embeddings from M2, enabling efficient binary retrieval using Hamming distance. This provides a fast retrieval baseline to compare against dense retrieval (M2).

## Architecture

```
Dense Embeddings (M2)
        ↓
BinaryQuantizer (threshold at 0)
        ↓
Binary Embeddings (1-bit per dim)
        ↓
Packed as bytes (8 bits/byte)
        ↓
BinaryRetriever (Hamming distance)
        ↓
Semantic Search (top-k by distance)
```

## Components

### BinaryQuantizer

Converts float embeddings to 1-bit binary.

```python
from src.efficient_rag.milestone3.quantizer import BinaryQuantizer

quantizer = BinaryQuantizer(dim=384)

# Quantize: threshold at 0
binary = quantizer.quantize(dense_embeddings)  # (N, 48) uint8

# Dequantize: reconstruct as 0/1
reconstructed = quantizer.dequantize(binary)   # (N, 384) float32
```

**Strategy**: Threshold at 0.
- Positive values → 1
- Negative values → 0
- Deterministic and efficient

**Packing**: 8 bits packed into each byte (uint8).
- Storage: 384 dims → 48 bytes per embedding
- Compression: 48x vs float32

### BinaryRetriever

Indexes and searches binary embeddings using Hamming distance.

```python
from src.efficient_rag.milestone3.binary_retriever import BinaryRetriever

retriever = BinaryRetriever(quantizer)
retriever.build_index(chunks, binary_embeddings)

results = retriever.search(query_binary, top_k=5)
# Returns: [{chunk_id, metadata, hamming_distance}, ...]
```

**Search**: Hamming distance (number of differing bits).
- Lower distance = more similar
- Exact match = 0
- Maximum = 384 bits

## Integration

```python
from src.efficient_rag.milestone1.document_processor import DocumentProcessor
from src.efficient_rag.milestone2.embedding import DenseEmbedder
from src.efficient_rag.milestone3.quantizer import BinaryQuantizer
from src.efficient_rag.milestone3.binary_retriever import BinaryRetriever

# M1: Load chunks
processor = DocumentProcessor()
chunks = processor.process_pipeline('data')

# M2: Dense embeddings
embedder = DenseEmbedder()
dense_embeddings = embedder.encode_chunks([c["text"] for c in chunks])

# M3: Quantize and search
quantizer = BinaryQuantizer(dim=384)
binary_embeddings = quantizer.quantize(dense_embeddings)
retriever = BinaryRetriever(quantizer)
retriever.build_index(chunks, binary_embeddings)

# Search
query_dense = embedder.encode_query("search term")
query_binary = quantizer.quantize(query_dense.reshape(1, -1))[0]
results = retriever.search(query_binary, top_k=5)
```

## Performance

See `examples/m3_binary_benchmark.py` for measured values.

- **Storage**: ~48x compression vs float32
- **Query speed**: Depends on implementation; Hamming distance is faster than inner product but more memory-bound
- **Recall@K**: Typically 85-95% of dense for well-distributed embeddings

## Testing

16 tests covering:
- Quantization correctness (shape, dtype, threshold)
- Bit packing/unpacking
- Hamming distance computation
- Binary retrieval and ranking
- Edge cases

## Design Notes

**Why 1-bit?**
- Extreme compression
- Fast Hamming distance computation
- Simple, deterministic thresholding

**Why threshold at 0?**
- Natural midpoint for normalized embeddings
- No calibration needed
- Preserves variance

**Hamming distance vs other similarities:**
- Inner product requires bit parity (complex)
- Hamming is simple and standard
- Trade-off: slightly slower than optimized implementations but correct and maintainable

## Limitations

- Resolution loss: 384-bit approximation of original continuous space
- Recall not always 100%: quantization adds error
- Bit packing overhead for small datasets

## Future Work (M4+)

- M4: Hybrid retrieval (dense + binary)
- M5: Approximate Hamming (e.g., locality-sensitive hashing)
- M6: Production indexing (GPU acceleration)

---

**Status**: M3 Complete | Tests: 16/16 passing
