"""
M2 Benchmark: Measure dense retrieval performance.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efficient_rag.milestone1.document_processor import DocumentProcessor
from efficient_rag.milestone2.embedding import DenseEmbedder
from efficient_rag.milestone2.retriever import DenseRetriever


def benchmark():
    """Run M2 performance benchmark."""
    
    print("\n" + "=" * 70)
    print("M2 BENCHMARK: Dense Retrieval with FAISS")
    print("=" * 70)
    
    # Load M1 chunks
    print("\n[1] Loading M1 chunks...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_pipeline("data/sample_documents", verbose=False)
    num_chunks = len(chunks)
    print(f"    Loaded: {num_chunks} chunks")
    
    # Initialize embedder
    print("\n[2] Initializing embedder...")
    embedder = DenseEmbedder()
    print(f"    Embedding dimension: {embedder.dim}")
    
    # Benchmark: Index building
    print("\n[3] BENCHMARK: Index building...")
    start = time.time()
    retriever = DenseRetriever(embedder)
    num_indexed = retriever.build_index(chunks)
    build_time = time.time() - start
    print(f"    Indexed: {num_indexed} chunks")
    print(f"    Time: {build_time:.3f}s")
    print(f"    Rate: {num_chunks / build_time:.1f} chunks/sec")
    
    # Benchmark: Query latency
    print("\n[4] BENCHMARK: Query latency...")
    queries = [
        "machine learning",
        "neural networks",
        "data processing"
    ]
    
    latencies = []
    for query in queries:
        start = time.time()
        results = retriever.search(query, top_k=5)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        print(f"    Query '{query}': {latency:.2f}ms ({len(results)} results)")
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"    Average latency: {avg_latency:.2f}ms")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Indexed chunks:     {num_indexed}")
    print(f"Embedding dim:      {embedder.dim}")
    print(f"Index build time:   {build_time:.3f}s")
    print(f"Query latency (avg): {avg_latency:.2f}ms")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    benchmark()
