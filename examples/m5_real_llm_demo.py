"""
M5 Real LLM Demo: End-to-end RAG pipeline with real OpenAI or Anthropic provider.

This demo shows how to run the complete M1-M5 pipeline with a real LLM provider
(OpenAI or Anthropic) instead of the mock provider used for testing.

SETUP:
1. Install dependencies: pip install -r requirements.txt
2. Create .env file from .env.example
3. Add your API key to .env (OPENAI_API_KEY or ANTHROPIC_API_KEY)
4. Set LLM_PROVIDER to 'openai' or 'anthropic'
5. Run: python examples/m5_real_llm_demo.py

Or set environment variables:
  export LLM_PROVIDER=openai
  export LLM_MODEL=gpt-4
  export OPENAI_API_KEY=sk-...
  python examples/m5_real_llm_demo.py

Windows CMD:
  set LLM_PROVIDER=openai
  set LLM_MODEL=gpt-4
  set OPENAI_API_KEY=sk-...
  python examples/m5_real_llm_demo.py
"""

import sys
import os
import time
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efficient_rag.milestone1.document_processor import DocumentProcessor
from efficient_rag.milestone2.embedding import DenseEmbedder
from efficient_rag.milestone2.retriever import DenseRetriever
from efficient_rag.milestone3.quantizer import BinaryQuantizer
from efficient_rag.milestone3.binary_retriever import BinaryRetriever
from efficient_rag.milestone4.hybrid_retriever import HybridRetriever
from efficient_rag.milestone5.llm_provider import get_llm_provider
from efficient_rag.milestone5.rag_generator import RAGGenerator


def main():
    """Run M5 end-to-end RAG demo with real LLM provider."""
    
    print("\n" + "=" * 90)
    print("M5 REAL LLM DEMO: End-to-End RAG Pipeline")
    print("=" * 90)
    
    # Check configuration
    llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    llm_model = os.environ.get("LLM_MODEL", "gpt-4" if llm_provider == "openai" else "claude-3-sonnet-20240229")
    
    print(f"\nConfiguration:")
    print(f"  LLM_PROVIDER: {llm_provider}")
    print(f"  LLM_MODEL: {llm_model}")
    print(f"  LLM_TEMPERATURE: {os.environ.get('LLM_TEMPERATURE', '0.2')}")
    
    # Verify API key is set
    if llm_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n❌ ERROR: OPENAI_API_KEY not set")
            print("   Set it in .env or export OPENAI_API_KEY=sk-...")
            return
        print(f"  OPENAI_API_KEY: set")
    elif llm_provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n❌ ERROR: ANTHROPIC_API_KEY not set")
            print("   Set it in .env or export ANTHROPIC_API_KEY=claude...")
            return
        print(f"  ANTHROPIC_API_KEY: set")
    
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
    print(f"✓ Built FAISS index")
    
    # M3: Binary quantization and Hamming index
    print("\n[M3] Building binary retrieval index...")
    quantizer = BinaryQuantizer(dim=384)
    texts = [c["text"] for c in chunks]
    binary_embeddings = quantizer.quantize(embedder.encode_chunks(texts))
    binary_retriever = BinaryRetriever(quantizer)
    binary_retriever.build_index(chunks, binary_embeddings)
    print(f"✓ Built binary index")
    
    # M4: Hybrid retrieval fusion
    print("\n[M4] Creating hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        dense_retriever, binary_retriever, quantizer, embedder,
        fusion_strategy="rrf"
    )
    print(f"✓ Hybrid retriever ready")
    
    # M5: Real LLM provider
    print("\n[M5] Initializing real LLM provider...")
    try:
        llm_provider_instance = get_llm_provider()
        print(f"✓ {llm_provider_instance.__class__.__name__} initialized")
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Create RAG generator
    rag = RAGGenerator(hybrid_retriever, llm_provider_instance, max_context_tokens=2000)
    print(f"✓ RAG generator ready")
    
    # Demo queries
    print("\n" + "=" * 90)
    print("DEMO QUERIES")
    print("=" * 90)
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are embeddings used for?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 90}")
        print(f"Query {i}: {query}")
        print('─' * 90)
        
        # Generate answer with timing
        start = time.time()
        try:
            result = rag.generate(query, top_k=3)
            elapsed = time.time() - start
            
            # Display results
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources: {result['sources']}")
            print(f"Chunks retrieved: {result['num_chunks_retrieved']}")
            print(f"Context sufficient: {result['has_sufficient_context']}")
            print(f"Latency: {elapsed:.2f}s")
            
        except RuntimeError as e:
            print(f"\n❌ ERROR: {e}")
            print("   Check your API key and rate limits")
            break
    
    print("\n" + "=" * 90)
    print("✓ M5 Real LLM Demo Complete")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
