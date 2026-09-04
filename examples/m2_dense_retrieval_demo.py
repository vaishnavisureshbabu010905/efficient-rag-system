"""
M2 Demo: Dense retrieval baseline using FAISS.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efficient_rag.milestone1.document_processor import DocumentProcessor
from efficient_rag.milestone2.embedding import DenseEmbedder
from efficient_rag.milestone2.retriever import DenseRetriever


def main():
    """Run M2 dense retrieval demo."""
    
    print("=" * 70)
    print("M2 DEMO: Dense Embedding + FAISS Retrieval")
    print("=" * 70)
    
    # Step 1: Load M1 chunks
    print("\n[1] Loading M1 chunks...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_pipeline("data/sample_documents", verbose=False)
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Step 2: Initialize embedder
    print("\n[2] Initializing embedder...")
    embedder = DenseEmbedder()
    print(f"✅ Embedding dimension: {embedder.dim}")
    
    # Step 3: Build index
    print("\n[3] Building FAISS index...")
    retriever = DenseRetriever(embedder)
    start = time.time()
    num_indexed = retriever.build_index(chunks)
    build_time = time.time() - start
    print(f"✅ Indexed {num_indexed} chunks in {build_time:.3f}s")
    
    # Step 4: Search
    queries = [
        "machine learning models",
        "data quality and preprocessing",
        "neural networks"
    ]
    
    print("\n[4] Running searches...")
    for query in queries:
        print(f"\n  Query: '{query}'")
        start = time.time()
        results = retriever.search(query, top_k=3)
        latency = (time.time() - start) * 1000  # ms
        
        for i, result in enumerate(results, 1):
            print(f"    [{i}] Score: {result['score']:.4f} | ID: {result['chunk_id']}")
            print(f"        Text: {result['text'][:80]}...")
        
        print(f"    Latency: {latency:.2f}ms")
    
    # Step 5: Save/load
    print("\n[5] Testing persistence...")
    index_dir = "data/outputs/m2_index"
    retriever.save_index(index_dir)
    print(f"✅ Saved index to {index_dir}")
    
    retriever2 = DenseRetriever(embedder)
    num_loaded = retriever2.load_index(index_dir)
    print(f"✅ Loaded {num_loaded} chunks")
    
    # Search after loading
    result = retriever2.search("machine learning", top_k=1)
    print(f"✅ Search after load: score={result[0]['score']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ M2 Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
