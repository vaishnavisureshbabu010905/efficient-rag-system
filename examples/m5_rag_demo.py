"""
M5 Demo: End-to-end RAG pipeline from document ingestion to answer generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efficient_rag.milestone1.document_processor import DocumentProcessor
from efficient_rag.milestone2.embedding import DenseEmbedder
from efficient_rag.milestone2.retriever import DenseRetriever
from efficient_rag.milestone3.quantizer import BinaryQuantizer
from efficient_rag.milestone3.binary_retriever import BinaryRetriever
from efficient_rag.milestone4.hybrid_retriever import HybridRetriever
from efficient_rag.milestone5.llm_provider import get_llm_provider
from efficient_rag.milestone5.rag_generator import RAGGenerator, QueryProcessor


def main():
    """Run M5 end-to-end RAG demo."""
    
    print("\n" + "=" * 90)
    print("M5 DEMO: End-to-End RAG Pipeline (M1 → M2 → M3 → M4 → M5)")
    print("=" * 90)
    
    # M1: Load documents
    print("\n[M1] Loading documents...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_pipeline("data/sample_documents", verbose=False)
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # M2: Dense embeddings and FAISS index
    print("\n[M2] Building dense retrieval index...")
    embedder = DenseEmbedder()
    dense_retriever = DenseRetriever(embedder)
    dense_retriever.build_index(chunks)
    print(f"✓ Built FAISS index with {len(chunks)} chunks")
    
    # M3: Binary quantization and Hamming index
    print("\n[M3] Building binary retrieval index...")
    quantizer = BinaryQuantizer(dim=384)
    texts = [c["text"] for c in chunks]
    binary_embeddings = quantizer.quantize(embedder.encode_chunks(texts))
    binary_retriever = BinaryRetriever(quantizer)
    binary_retriever.build_index(chunks, binary_embeddings)
    print(f"✓ Built binary index (48x compression)")
    
    # M4: Hybrid retrieval fusion
    print("\n[M4] Creating hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        dense_retriever, binary_retriever, quantizer, embedder,
        fusion_strategy="rrf"
    )
    print(f"✓ Hybrid retriever ready (RRF fusion)")
    
    # M5: RAG generation
    print("\n[M5] Initializing RAG generator...")
    llm_provider = get_llm_provider()  # Defaults to mock for demo
    rag = RAGGenerator(hybrid_retriever, llm_provider, max_context_tokens=2000)
    print(f"✓ RAG generator ready (LLM: {llm_provider.__class__.__name__})")
    
    # Demo queries
    print("\n" + "=" * 90)
    print("DEMO QUERIES")
    print("=" * 90)
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the applications of deep learning?"
    ]
    
    for query in queries:
        print(f"\n{'─' * 90}")
        print(f"Query: {query}")
        print('─' * 90)
        
        # Preprocess query
        processed_query = QueryProcessor.preprocess(query)
        
        # Generate answer
        result = rag.generate(processed_query, top_k=3)
        
        # Display results
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources ({result['num_chunks_retrieved']} chunks):")
        for i, source in enumerate(result['sources'], 1):
            chunk = next((c for c in chunks if c['metadata']['chunk_id'] == source), None)
            if chunk:
                print(f"  [{i}] {source}: {chunk['text'][:80]}...")
        
        print(f"\nContext Sufficient: {'Yes' if result['has_sufficient_context'] else 'No'}")
    
    print("\n" + "=" * 90)
    print("✓ M5 Demo Complete")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
