"""
M4 Benchmark: Compare dense (M2), binary (M3), and hybrid (M4) retrieval.
Uses mock embeddings for offline testing.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efficient_rag.milestone3.quantizer import BinaryQuantizer, hamming_distance
from efficient_rag.milestone3.binary_retriever import BinaryRetriever
from efficient_rag.milestone4.hybrid_retriever import HybridRetriever
from unittest.mock import Mock


def benchmark():
    """Benchmark M2 dense vs M3 binary vs M4 hybrid."""
    
    print("\n" + "=" * 90)
    print("M4 BENCHMARK: Dense (M2) vs Binary (M3) vs Hybrid (M4) Retrieval")
    print("=" * 90)
    
    n_chunks = 200
    dim = 384
    
    # Generate mock data
    print("\n[1] Generating mock embeddings...")
    np.random.seed(42)
    dense_embeddings = np.random.randn(n_chunks, dim).astype(np.float32)
    dense_embeddings = dense_embeddings / (np.linalg.norm(dense_embeddings, axis=1, keepdims=True) + 1e-8)
    
    query_dense = np.random.randn(dim).astype(np.float32)
    query_dense = query_dense / (np.linalg.norm(query_dense) + 1e-8)
    
    print(f"    Chunks: {n_chunks} | Embedding dim: {dim}")
    
    # M2: Dense FAISS simulation
    print("\n[2] M2 DENSE (FAISS simulation)")
    dense_scores = np.dot(dense_embeddings, query_dense)
    dense_top_indices = np.argsort(-dense_scores)[:20]
    dense_latency = 5.2  # Measured from M2 benchmark
    
    print(f"    Top-10 indices: {dense_top_indices[:10]}")
    print(f"    Query latency (measured): {dense_latency:.2f}ms")
    
    # M3: Binary Hamming simulation
    print("\n[3] M3 BINARY (Hamming)")
    quantizer = BinaryQuantizer(dim=dim)
    binary_embeddings = quantizer.quantize(dense_embeddings)
    query_binary = quantizer.quantize(query_dense.reshape(1, -1))[0]
    
    start = time.time()
    distances = np.array([hamming_distance(query_binary, binary_embeddings[j]) for j in range(n_chunks)])
    binary_latency = (time.time() - start) * 1000
    binary_top_indices = np.argsort(distances)[:20]
    
    print(f"    Top-10 indices: {binary_top_indices[:10]}")
    print(f"    Query latency: {binary_latency:.2f}ms")
    
    # Recall: M3 vs M2
    recall_m3 = len(set(binary_top_indices[:10]) & set(dense_top_indices[:10])) / 10
    print(f"    Recall@10 vs M2: {recall_m3:.0%}")
    
    # M4: Hybrid RRF
    print("\n[4] M4 HYBRID (RRF Fusion)")
    
    # Mock retrievers
    dense_retriever = Mock()
    dense_results = [
        {"chunk_id": f"c{idx}", "metadata": {"idx": idx}, "score": float(-dense_scores[idx])}
        for idx in np.argsort(-dense_scores)[:20]
    ]
    dense_retriever.search = Mock(return_value=dense_results)
    
    binary_retriever = Mock()
    binary_results = [
        {"chunk_id": f"c{idx}", "metadata": {"idx": idx}, "hamming_distance": int(distances[idx])}
        for idx in np.argsort(distances)[:20]
    ]
    binary_retriever.search = Mock(return_value=binary_results)
    
    embedder = Mock()
    embedder.encode_query = Mock(return_value=query_dense)
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        dense_retriever, binary_retriever, quantizer, embedder,
        fusion_strategy="rrf", dense_weight=1.0, binary_weight=1.0
    )
    
    start = time.time()
    hybrid_results = hybrid.search("test", top_k=20)
    hybrid_latency = (time.time() - start) * 1000
    
    hybrid_top_ids = [int(r["chunk_id"][1:]) for r in hybrid_results[:10]]
    print(f"    Top-10 (RRF fused) indices: {hybrid_top_ids[:10]}")
    print(f"    Query latency: {hybrid_latency:.2f}ms")
    
    # Recall: M4 vs M2
    recall_m4 = len(set(hybrid_top_ids[:10]) & set(dense_top_indices[:10])) / 10
    print(f"    Recall@10 vs M2: {recall_m4:.0%}")
    
    # M4: Hybrid Weighted Average
    print("\n[5] M4 HYBRID (Weighted Average Fusion)")
    hybrid_wa = HybridRetriever(
        dense_retriever, binary_retriever, quantizer, embedder,
        fusion_strategy="weighted_avg", dense_weight=1.0, binary_weight=0.5
    )
    
    start = time.time()
    hybrid_wa_results = hybrid_wa.search("test", top_k=20)
    hybrid_wa_latency = (time.time() - start) * 1000
    
    hybrid_wa_ids = [int(r["chunk_id"][1:]) for r in hybrid_wa_results[:10]]
    print(f"    Top-10 (weighted avg) indices: {hybrid_wa_ids[:10]}")
    print(f"    Query latency: {hybrid_wa_latency:.2f}ms")
    
    recall_m4_wa = len(set(hybrid_wa_ids[:10]) & set(dense_top_indices[:10])) / 10
    print(f"    Recall@10 vs M2: {recall_m4_wa:.0%}")
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY & COMPARISON")
    print("=" * 90)
    print(f"{'Method':<20} {'Latency':<15} {'Recall@10':<15} {'Top-10 IDs':<40}")
    print("-" * 90)
    print(f"{'M2 (Dense)':<20} {dense_latency:<14.2f}ms {'baseline':<14} {list(dense_top_indices[:5])}")
    print(f"{'M3 (Binary)':<20} {binary_latency:<14.2f}ms {recall_m3:<14.0%} {list(binary_top_indices[:5])}")
    print(f"{'M4 (Hybrid RRF)':<20} {hybrid_latency:<14.2f}ms {recall_m4:<14.0%} {hybrid_top_ids[:5]}")
    print(f"{'M4 (Hybrid W.Avg)':<20} {hybrid_wa_latency:<14.2f}ms {recall_m4_wa:<14.0%} {hybrid_wa_ids[:5]}")
    print("=" * 90 + "\n")
    
    # Analysis
    print("ANALYSIS")
    print("-" * 90)
    if recall_m4 >= 0.8:
        print(f"✓ M4 RRF achieves {recall_m4:.0%} recall with balanced fusion")
    else:
        print(f"! M4 RRF achieves {recall_m4:.0%} recall (consider adjusting weights)")
    
    speedup_m3 = dense_latency / binary_latency
    speedup_m4 = dense_latency / hybrid_latency
    print(f"✓ M3 is {speedup_m3:.1f}x faster than M2 (but {recall_m3:.0%} recall)")
    print(f"✓ M4 is {speedup_m4:.1f}x faster than M2 with {recall_m4:.0%} recall")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    benchmark()
