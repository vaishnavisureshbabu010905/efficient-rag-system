"""Tests for M5: RAG Generation."""

import pytest
from unittest.mock import Mock
from src.efficient_rag.milestone5.llm_provider import (
    MockLLMProvider, get_llm_provider, LLMProvider
)
from src.efficient_rag.milestone5.rag_generator import RAGGenerator, QueryProcessor


class TestMockLLMProvider:
    """Test mock LLM provider."""
    
    def test_init(self):
        """Test initialization."""
        provider = MockLLMProvider(model_name="test-model")
        assert provider.model_name == "test-model"
    
    def test_generate_mock_response(self):
        """Test mock generation."""
        provider = MockLLMProvider()
        response = provider.generate("test prompt")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_with_answer_section(self):
        """Test mock generation with Answer: section."""
        provider = MockLLMProvider()
        prompt = "Context: test\nAnswer:"
        response = provider.generate(prompt)
        
        assert "[Generated answer" in response or "[Mock response]" in response


class TestLLMProviderFactory:
    """Test LLM provider factory."""
    
    def test_get_mock_provider_default(self):
        """Test mock provider when explicitly selected."""
        # Default is now openai, so explicitly select mock
        provider = get_llm_provider(provider="mock")
        assert isinstance(provider, MockLLMProvider)
    
    def test_get_mock_provider_explicit(self):
        """Test explicit mock provider."""
        provider = get_llm_provider(provider="mock", model="custom-mock")
        assert isinstance(provider, MockLLMProvider)
        assert provider.model_name == "custom-mock"
    
    def test_openai_without_api_key_raises(self):
        """Test OpenAI provider requires API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm_provider(provider="openai")
    
    def test_anthropic_without_api_key_raises(self):
        """Test Anthropic provider requires API key."""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_llm_provider(provider="anthropic")


class TestQueryProcessor:
    """Test query processing utilities."""
    
    def test_preprocess_whitespace(self):
        """Test whitespace normalization."""
        query = "  hello    world  "
        result = QueryProcessor.preprocess(query)
        assert result == "hello world"
    
    def test_preprocess_preserves_punctuation(self):
        """Test punctuation is preserved."""
        query = "What is this?"
        result = QueryProcessor.preprocess(query)
        assert result == "What is this?"
    
    def test_expand_query_basic(self):
        """Test query expansion."""
        query = "machine learning"
        variants = QueryProcessor.expand_query(query)
        
        assert len(variants) >= 1
        assert query in variants
    
    def test_expand_query_with_question(self):
        """Test query expansion with question mark."""
        query = "What is machine learning?"
        variants = QueryProcessor.expand_query(query)
        
        assert query in variants


class TestRAGGenerator:
    """Test RAG generator."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Mock hybrid retriever."""
        retriever = Mock()
        retriever.search = Mock(return_value=[
            {
                "chunk_id": "c0",
                "metadata": {"chunk_id": "c0", "filename": "f0"},
                "text": "Machine learning is a type of artificial intelligence.",
                "fusion_score": 0.9
            },
            {
                "chunk_id": "c1",
                "metadata": {"chunk_id": "c1", "filename": "f1"},
                "text": "Deep learning uses neural networks with multiple layers.",
                "fusion_score": 0.85
            }
        ])
        return retriever
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM provider."""
        llm = Mock(spec=LLMProvider)
        llm.generate = Mock(return_value="Machine learning is an AI technique that learns from data.")
        return llm
    
    @pytest.fixture
    def rag_generator(self, mock_retriever, mock_llm):
        """Create RAG generator with mocks."""
        return RAGGenerator(mock_retriever, mock_llm)
    
    def test_init(self, rag_generator):
        """Test initialization."""
        assert rag_generator.retriever is not None
        assert rag_generator.llm is not None
        assert rag_generator.max_context_tokens > 0
    
    def test_generate_basic(self, rag_generator):
        """Test basic answer generation."""
        result = rag_generator.generate("What is machine learning?")
        
        assert "query" in result
        assert "answer" in result
        assert "sources" in result
        assert "retrieval_results" in result
        assert "has_sufficient_context" in result
    
    def test_generate_has_sources(self, rag_generator):
        """Test generated result includes sources."""
        result = rag_generator.generate("What is machine learning?")
        
        assert len(result["sources"]) > 0
        assert "c0" in result["sources"]
    
    def test_generate_has_sufficient_context(self, rag_generator):
        """Test sufficient context detection."""
        result = rag_generator.generate("What is ML?")
        
        assert result["has_sufficient_context"] is True
        assert result["num_chunks_retrieved"] > 0
    
    def test_generate_no_context(self, mock_llm):
        """Test generation with no retrieved context."""
        retriever = Mock()
        retriever.search = Mock(return_value=[])
        
        rag = RAGGenerator(retriever, mock_llm)
        result = rag.generate("obscure question")
        
        assert result["has_sufficient_context"] is False
        assert len(result["sources"]) == 0
    
    def test_generate_with_score_threshold(self, mock_retriever, mock_llm):
        """Test filtering by fusion score."""
        rag = RAGGenerator(mock_retriever, mock_llm, min_chunk_score=0.9)
        result = rag.generate("test")
        
        # Only c0 has score >= 0.9
        assert len(result["sources"]) <= 1
    
    def test_context_assembly(self, rag_generator):
        """Test context is properly assembled."""
        result = rag_generator.generate("test")
        
        # Check LLM was called with prompt containing context
        rag_generator.llm.generate.assert_called_once()
        call_args = rag_generator.llm.generate.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt")
        
        # Prompt should contain context text
        assert "artificial intelligence" in prompt or "Context:" in prompt
    
    def test_stream_generate(self, rag_generator):
        """Test streaming generation."""
        stream = rag_generator.stream_generate("test", top_k=2)
        results = list(stream)
        
        assert len(results) >= 1
        assert "answer" in results[0]
    
    def test_generate_top_k_parameter(self, mock_retriever, mock_llm):
        """Test top_k parameter is passed to retriever."""
        rag = RAGGenerator(mock_retriever, mock_llm)
        rag.generate("test", top_k=3)
        
        mock_retriever.search.assert_called()
        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs.get("top_k") == 3


class TestRAGIntegration:
    """Integration tests for RAG pipeline."""
    
    def test_full_pipeline(self):
        """Test full RAG pipeline with mocks."""
        # Mock retriever
        retriever = Mock()
        retriever.search = Mock(return_value=[
            {
                "chunk_id": "c0",
                "metadata": {"chunk_id": "c0"},
                "text": "Answer text",
                "fusion_score": 0.9
            }
        ])
        
        # Mock LLM
        llm = Mock()
        llm.generate = Mock(return_value="Final answer")
        
        # Create RAG
        rag = RAGGenerator(retriever, llm)
        
        # Generate
        result = rag.generate("question")
        
        # Verify pipeline
        assert result["query"] == "question"
        assert result["answer"] == "Final answer"
        assert result["sources"] == ["c0"]
        
        # Verify retriever was called
        retriever.search.assert_called_once()
        
        # Verify LLM was called
        llm.generate.assert_called_once()
    
    def test_query_preprocessing_integration(self):
        """Test query preprocessing in context."""
        processed = QueryProcessor.preprocess("  What is AI?  ")
        assert processed == "What is AI?"
        
        variants = QueryProcessor.expand_query(processed)
        assert len(variants) >= 1
