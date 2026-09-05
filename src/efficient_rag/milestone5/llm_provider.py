"""
LLM Provider abstraction for M5: Support multiple LLM backends via environment variables.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt."""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM for testing (no API calls)."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate mock response."""
        # Simple mock: extract "Answer:" section if present
        if "Answer:" in prompt:
            return "[Generated answer based on context - this is a mock response]"
        return "[Mock response]"


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider (requires API key)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        try:
            import openai
            openai.api_key = self.api_key
            self.openai = openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude provider (requires API key)."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Anthropic Claude."""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


def get_llm_provider(provider: Optional[str] = None, model: Optional[str] = None, 
                     api_key: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get LLM provider from environment or explicit parameters.
    
    Priority:
    1. Explicit parameters
    2. Environment variables: LLM_PROVIDER, LLM_MODEL, LLM_API_KEY
    3. Defaults to MockLLMProvider
    
    Args:
        provider: Provider name (openai, anthropic, mock)
        model: Model name/identifier
        api_key: API key (if needed)
        
    Returns:
        LLMProvider instance
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "mock").lower()
    
    if provider == "openai":
        model = model or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        key = api_key or os.environ.get("LLM_API_KEY")
        return OpenAILLMProvider(model_name=model, api_key=key)
    
    elif provider == "anthropic":
        model = model or os.environ.get("LLM_MODEL", "claude-3-sonnet-20240229")
        key = api_key or os.environ.get("LLM_API_KEY")
        return AnthropicLLMProvider(model_name=model, api_key=key)
    
    else:  # mock or unknown
        model = model or os.environ.get("LLM_MODEL", "mock-model")
        return MockLLMProvider(model_name=model)
