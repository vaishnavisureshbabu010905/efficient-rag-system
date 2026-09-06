"""
RAG Generator for M5: Retrieval-Augmented Generation with grounded answers.
"""

from typing import List, Dict, Any, Optional, Tuple
from .llm_provider import LLMProvider


class RAGGenerator:
    """End-to-end RAG pipeline: retrieval → context assembly → generation."""
    
    def __init__(self, hybrid_retriever, llm_provider: LLMProvider,
                 max_context_tokens: int = 2000,
                 min_chunk_score: Optional[float] = None,
                 context_window: int = 4096):
        """
        Initialize RAG generator.
        
        Args:
            hybrid_retriever: M4 HybridRetriever instance
            llm_provider: LLM provider (M5)
            max_context_tokens: Max tokens for context assembly (rough estimate)
            min_chunk_score: Minimum score threshold for retrieved chunks
            context_window: Total context window for prompt (for safety)
        """
        self.retriever = hybrid_retriever
        self.llm = llm_provider
        self.max_context_tokens = max_context_tokens
        self.min_chunk_score = min_chunk_score
        self.context_window = context_window
    
    def generate(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate grounded answer for query using hybrid retrieval.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with:
            - query: Original query
            - answer: Generated answer
            - sources: List of chunk IDs used
            - retrieval_results: Raw retrieval data
            - has_sufficient_context: Bool indicating if context was adequate
        """
        # Step 1: Retrieve
        retrieval_results = self.retriever.search(query, top_k=top_k)
        
        # Step 2: Filter by score if threshold set
        if self.min_chunk_score is not None:
            retrieval_results = [
                r for r in retrieval_results 
                if r.get("fusion_score", 0) >= self.min_chunk_score
            ]
        
        # Step 3: Check if context is sufficient
        has_sufficient_context = len(retrieval_results) > 0
        
        # Step 4: Assemble context
        context_text = self._assemble_context(retrieval_results)
        
        # Step 5: Generate prompt
        prompt = self._build_prompt(query, context_text, has_sufficient_context)
        
        # Step 6: Generate answer
        answer = self.llm.generate(prompt, max_tokens=512)
        
        # Step 7: Extract and return
        sources = [r["chunk_id"] for r in retrieval_results]
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieval_results": retrieval_results,
            "has_sufficient_context": has_sufficient_context,
            "num_chunks_retrieved": len(retrieval_results)
        }
    
    def _assemble_context(self, retrieval_results: List[Dict[str, Any]]) -> str:
        """Assemble context from retrieved chunks."""
        if not retrieval_results:
            return "[No relevant context found]"
        
        context_parts = []
        token_count = 0
        
        for i, result in enumerate(retrieval_results, 1):
            chunk_text = result.get("metadata", {}).get("chunk_id", f"Chunk {i}")
            text = f"[{chunk_text}]\n{result.get('text', '')}"
            
            # Rough token estimate (1 token ≈ 4 chars)
            text_tokens = len(text) // 4
            
            if token_count + text_tokens > self.max_context_tokens:
                break
            
            context_parts.append(text)
            token_count += text_tokens
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, has_context: bool) -> str:
        """Build RAG prompt with query and context."""
        if not has_context:
            return f"""You are a helpful assistant. Answer the following question based ONLY on the provided context.

Question: {query}

Unfortunately, no relevant context was found in the knowledge base.
Please respond with: "I do not have enough context to answer this question."
Do NOT use outside knowledge. Do NOT make up information."""
        
        return f"""You are a helpful assistant. Answer the following question ONLY using the provided context.

IMPORTANT RULES:
1. Answer ONLY from the context provided below.
2. Do NOT use outside knowledge or make assumptions.
3. If the context does not contain the answer, explicitly state: "I do not have enough context to answer this."
4. Be direct, clear, and concise.

Context from knowledge base:
{context}

Question: {query}

Answer:"""
    
    def stream_generate(self, query: str, top_k: int = 5):
        """
        Generate answer and yield results (for streaming if needed).
        For now, yields single result (can be extended for actual streaming).
        """
        result = self.generate(query, top_k)
        yield result


class QueryProcessor:
    """Simple query preprocessing."""
    
    @staticmethod
    def preprocess(query: str) -> str:
        """Preprocess query (basic cleaning)."""
        # Remove extra whitespace
        query = " ".join(query.split())
        # Remove trailing punctuation for cleaner retrieval
        if query and query[-1] in ".!?":
            # Keep it for now, as it may be useful for LLM
            pass
        return query
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """Generate query variants for multi-attempt retrieval."""
        # Simple expansion: original + question form
        variants = [query]
        
        # Add question form if not already a question
        if not query.endswith("?"):
            variants.append(f"{query}?")
        
        return variants
