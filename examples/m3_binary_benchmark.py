"""
M3 Benchmark: Compare dense (M2) vs binary (M3) retrieval.
Uses mock embeddings to avoid model downloads.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efficient_rag.milestone3.quantizer import BinaryQuantizer, hamming_distance
from efficient_rag.milestone3.binary_retriever import BinaryRetriever


def benchmark():
    """Benchmark M3 binary retrieval with mock data."""
    
    print("\n" + "=" * 80)
    print("M3 BENCHMARK: Binary Quantization & Hamming Retrieval (Mock Data)")
    print("=" * 80)
    
    n_chunks = 100
    dim = 384
    
    # Generate mock embeddings
    print("\n[1] Generating mock embeddings...")
    np.random.seed(42)
    dense_embeddings = np.random.randn(n_chunks, dim).astype(np.float32)
    dense_embeddings = dense_embeddings / (np.linalg.norm(dense_embeddings, axis=1, keepdims=True) + 1e-8)
    
    query_dense = np.random.randn(dim).astype(np.float32)
    query_dense = query_dense / (np.linalg.norm(query_dense) + 1e-8)
    
    print(f"    Dense embeddings: {dense_embeddings.shape}")
    print(f"    Query: {query_dense.shape}")
    
    # M3: Binary quantization
    print("\n[2] M3: BINARY QUANTIZATION")
    quantizer = BinaryQuantizer(dim=dim)
    
    start = time.time()
    binary_embeddings = quantizer.quantize(dense_embeddings)
    quant_time = time.time() - start
    
    dense_size = dense_embeddings.nbytes / 1024 / 1024  # MB
    binary_size = binary_embeddings.nbytes / 1024 / 1024  # MB
    compression = dense_size / binary_size
    
    print(f"    Dense size: {dense_size:.2f} MB")
    print(f"    Binary size: {binary_size:.2f} MB")
    print(f"    Compression: {compression:.1f}x")
    print(f"    Quantization time: {quant_time:.3f}s")
    
    # Search
    print("\n[3] SEARCH PERFORMANCE")
    query_binary = quantizer.quantize(query_dense.reshape(1, -1))[0]
    
    latencies = []
    for i in range(10):
        t0 = time.time()
        distances = np.array([hamming_distance(query_binary, binary_embeddings[j]) for j in range(n_chunks)])
        latencies.append((time.time() - t0) * 1000)
    
    avg_latency = np.mean(latencies)
    print(f"    10 queries on {n_chunks} chunks")
    print(f"    Average latency per query: {avg_latency:.2f}ms")
    print(f"    Throughput: {n_chunks / (avg_latency / 1000):.0f} chunks/sec")
    
    # Recall simulation
    print("\n[4] RECALL ANALYSIS")
    top_k = 10
    top_indices = np.argsort(distances)[:top_k]
    
    # Compute dense similarity for comparison
    dense_scores = np.dot(dense_embeddings, query_dense)
    dense_top_indices = np.argsort(-dense_scores)[:top_k]
    
    overlap = len(set(top_indices) & set(dense_top_indices))
    recall = overlap / top_k
    print(f"    Recall@{top_k}: {recall:.0%}")
    print(f"    Top-k overlap: {overlap}/{top_k} chunks match")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dense embedding: {dense_size:.2f} MB (float32)")
    print(f"Binary index:    {binary_size:.2f} MB (packed uint8)")
    print(f"Storage saving:  {compression:.1f}x compression")
    print(f"Query latency:   {avg_latency:.2f}ms per query")
    print(f"Recall@{top_k}:     {recall:.0%}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    benchmark()
