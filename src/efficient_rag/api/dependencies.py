"""
Dependencies for M6 API: initialize RAG pipeline.
"""

import os
from pathlib import Path
from functools import lru_cache
from src.efficient_rag.milestone1.document_processor import DocumentProcessor
from src.efficient_rag.milestone2.embedding import DenseEmbedder
from src.efficient_rag.milestone2.retriever import DenseRetriever
from src.efficient_rag.milestone3.quantizer import BinaryQuantizer
from src.efficient_rag.milestone3.binary_retriever import BinaryRetriever
from src.efficient_rag.milestone4.hybrid_retriever import HybridRetriever
from src.efficient_rag.milestone5.llm_provider import get_llm_provider
from src.efficient_rag.milestone5.rag_generator import RAGGenerator


class RAGPipeline:
    """Encapsulates the complete M1-M5 RAG pipeline."""
    
    def __init__(self, documents_path: str = "data/sample_documents"):
        self.documents_path = documents_path
        self._rag_generator = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components M1-M5."""
        try:
            # M1: Load documents
            if not Path(self.documents_path).exists():
                raise FileNotFoundError(f"Documents path not found: {self.documents_path}")
            
            processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
            chunks = processor.process_pipeline(self.documents_path, verbose=False)
            
            if not chunks:
                raise ValueError(f"No chunks loaded from {self.documents_path}")
            
            # M2: Dense retrieval
            embedder = DenseEmbedder()
            dense_retriever = DenseRetriever(embedder)
            dense_retriever.build_index(chunks)
            
            # M3: Binary retrieval
            quantizer = BinaryQuantizer(dim=384)
            texts = [c["text"] for c in chunks]
            binary_embeddings = quantizer.quantize(embedder.encode_chunks(texts))
            binary_retriever = BinaryRetriever(quantizer)
            binary_retriever.build_index(chunks, binary_embeddings)
            
            # M4: Hybrid retrieval
            hybrid_retriever = HybridRetriever(
                dense_retriever, binary_retriever, quantizer, embedder,
                fusion_strategy="rrf", dense_weight=1.0, binary_weight=1.0
            )
            
            # M5: RAG generation
            llm_provider = get_llm_provider()
            self._rag_generator = RAGGenerator(
                hybrid_retriever=hybrid_retriever,
                llm_provider=llm_provider,
                max_context_tokens=int(os.environ.get("MAX_CONTEXT_TOKENS", "2000")),
                min_chunk_score=None
            )
        
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RAG pipeline: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> dict:
        """Execute a query through the full pipeline."""
        if not self._rag_generator:
            raise RuntimeError("Pipeline not initialized")
        
        return self._rag_generator.generate(query_text, top_k=top_k)


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline (singleton)."""
    documents_path = os.environ.get("DOCUMENTS_PATH", "data/sample_documents")
    return RAGPipeline(documents_path=documents_path)
