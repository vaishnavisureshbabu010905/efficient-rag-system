"""Tests for production LLM providers (M5 upgrade)."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.efficient_rag.milestone5.llm_provider import (
    OpenAILLMProvider, AnthropicLLMProvider, GroqLLMProvider, MockLLMProvider, get_llm_provider
)


class TestOpenAILLMProvider:
    """Test OpenAI provider with modern SDK."""
    
    def test_init_requires_api_key(self):
        """Test OpenAI provider requires API key."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAILLMProvider()
    

    



class TestAnthropicLLMProvider:
    """Test Anthropic provider with modern SDK."""
    
    def test_init_requires_api_key(self):
        """Test Anthropic provider requires API key."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicLLMProvider()


class TestGroqLLMProvider:
    """Test Groq provider with OpenAI-compatible API."""
    
    def test_init_requires_api_key(self):
        """Test Groq provider requires API key."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                GroqLLMProvider()


class TestGetLLMProviderFactory:
    """Test LLM provider factory function."""
    
    def test_explicit_mock_provider(self):
        """Test explicit mock provider selection."""
        provider = get_llm_provider(provider="mock")
        assert isinstance(provider, MockLLMProvider)
    
    def test_invalid_provider_raises(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider(provider="invalid")
    
    def test_openai_missing_key_raises(self):
        """Test OpenAI without key raises error (no fallback to mock)."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_llm_provider(provider="openai")
    
    def test_anthropic_missing_key_raises(self):
        """Test Anthropic without key raises error (no fallback to mock)."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                get_llm_provider(provider="anthropic")
    
    def test_groq_missing_key_explicit_error(self):
        """Test Groq explicit error when key missing."""
        try:
            GroqLLMProvider()
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "GROQ_API_KEY" in str(e)
    
    def test_groq_missing_key_raises(self):
        """Test Groq without key raises error (no fallback to mock)."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                get_llm_provider(provider="groq")


class TestMockProviderStillWorks:
    """Ensure mock provider still works for testing."""
    
    def test_mock_provider_generate(self):
        """Test mock provider still generates without API calls."""
        provider = MockLLMProvider()
        result = provider.generate("test")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mock_provider_explicit(self):
        """Test explicit mock provider selection."""
        provider = get_llm_provider(provider="mock")
        assert isinstance(provider, MockLLMProvider)
        result = provider.generate("test")
        assert isinstance(result, str)
