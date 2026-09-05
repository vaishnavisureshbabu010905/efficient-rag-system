# Milestone 4: Hybrid Retrieval (Dense + Binary)

## Overview

M4 implements hybrid retrieval combining:
- **M2 Dense (FAISS)**: Exact cosine similarity search
- **M3 Binary (Hamming)**: Fast approximate search

Results are fused using configurable strategies to balance recall and latency.

## Architecture

```
Query
  ├─→ M2 Dense Retrieval ─→ Dense Results (scores)
  │
  ├─→ M3 Binary Retrieval ─→ Binary Results (Hamming distances)
  │
  └─→ HybridRetriever (Fusion) ─→ Fused Top-K Results
```

## Components

### HybridRetriever

Fuses results from M2 and M3 using configurable strategies.

```python
from src.efficient_rag.milestone4.hybrid_retriever import HybridRetriever

hybrid = HybridRetriever(
    dense_retriever=m2_retriever,
    binary_retriever=m3_retriever,
    quantizer=quantizer,
    embedding=embedder,
    fusion_strategy="rrf",  # or "weighted_avg"
    dense_weight=1.0,
    binary_weight=1.0,
    k=60  # RRF constant
)

results = hybrid.search("query", top_k=5)
```

**Fusion Strategies**:

### Reciprocal Rank Fusion (RRF)

```
score(d) = Σ (1 / (k + rank_i(d)))
```

- **k**: Constant (default 60, higher = more balanced)
- Treats both rankers equally by default
- Configurable weights for asymmetric importance

**Pros**: Rank-based, doesn't assume score distributions  
**Cons**: Ignores actual score values

### Weighted Average

```
score(d) = (norm_dense(d) × w_dense + norm_binary(d) × w_binary) / (w_dense + w_binary)
```

- Normalizes dense scores by max
- Inverts Hamming distance (lower = higher score)
- Linearly combines with weights

**Pros**: Uses score magnitudes, faster  
**Cons**: Assumes score scales are comparable

## Integration

```python
from src.efficient_rag.milestone1.document_processor import DocumentProcessor
from src.efficient_rag.milestone2.embedding import DenseEmbedder
from src.efficient_rag.milestone2.retriever import DenseRetriever
from src.efficient_rag.milestone3.quantizer import BinaryQuantizer
from src.efficient_rag.milestone3.binary_retriever import BinaryRetriever
from src.efficient_rag.milestone4.hybrid_retriever import HybridRetriever

# M1: Load chunks
processor = DocumentProcessor()
chunks = processor.process_pipeline('data')

# M2: Dense index
embedder = DenseEmbedder()
dense_retriever = DenseRetriever(embedder)
dense_retriever.build_index(chunks)

# M3: Binary index
quantizer = BinaryQuantizer(dim=384)
binary_embeddings = quantizer.quantize(
    embedder.encode_chunks([c["text"] for c in chunks])
)
binary_retriever = BinaryRetriever(quantizer)
binary_retriever.build_index(chunks, binary_embeddings)

# M4: Hybrid retriever
hybrid = HybridRetriever(
    dense_retriever, binary_retriever, quantizer, embedder,
    fusion_strategy="rrf"
)

# Search
results = hybrid.search("query", top_k=5)
for r in results:
    print(f"Rank {r['rank']}: {r['chunk_id']} (score: {r['fusion_score']:.4f})")
```

## Performance

See `examples/m4_hybrid_benchmark.py` for measured values on sample data.

**Trade-offs**:
- M2 (Dense): Best recall, baseline latency
- M3 (Binary): Fast, reduced recall (~50%)
- M4 (Hybrid RRF): Balanced recall (70-90%), minimal latency overhead

**Tuning**:
- **RRF weights**: Increase `dense_weight` to favor exact results
- **RRF k parameter**: Lower k = more emphasis on top ranks
- **Fusion strategy**: RRF for robustness, weighted avg for speed

## Testing

12 tests covering:
- RRF fusion correctness
- Weighted average fusion
- Result ranking and combining
- Weight effects on ranking
- Edge cases (empty results, duplicates)

## Design Notes

**Why two fusion strategies?**
- RRF is rank-based, works with any scoring system
- Weighted avg is faster, uses score magnitudes
- Both have tradeoffs in sensitivity to score distributions

**Configurable weights:**
- Allow prioritizing dense accuracy or binary speed
- Useful for different recall/latency requirements

**Modular design:**
- HybridRetriever doesn't depend on M2/M3 implementation details
- Works with any dense/binary retrievers
- Easy to add new fusion strategies

## Limitations

- Requires both indexes in memory
- Query must run both paths (no fallback optimization)
- Fusion parameter tuning is domain-specific

## Future Work (M5+)

- M5: Reranking with cross-encoders
- M6: Production deployment (FastAPI)
- M7: Multi-stage ranking (fast filters → slow rankers)

---

**Status**: M4 Complete | Tests: 12/12 passing | M1+M2+M3+M4 Total: 62/62
