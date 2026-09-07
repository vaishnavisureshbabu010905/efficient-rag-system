"""
LLM Provider abstraction for M5: Support multiple LLM backends via environment variables.
Production-ready implementation with real OpenAI/Anthropic providers.
"""

import os
import time
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
        if "Answer:" in prompt:
            return "[Generated answer based on context - this is a mock response]"
        return "[Mock response]"


class OpenAILLMProvider(LLMProvider):
    """Production OpenAI LLM provider using modern SDK (v1.0+)."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None,
                 temperature: float = 0.2, max_tokens: int = 512, timeout: float = 30.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it in .env or environment before using OpenAI provider."
            )
        
        try:
            from openai import OpenAI, APIError, RateLimitError, APIConnectionError
            self.client = OpenAI(api_key=self.api_key, timeout=timeout)
            self.APIError = APIError
            self.RateLimitError = RateLimitError
            self.APIConnectionError = APIConnectionError
        except ImportError:
            raise ImportError("openai>=1.0.0 required: pip install openai")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using OpenAI API."""
        max_tokens = max_tokens or self.max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except self.RateLimitError:
            raise RuntimeError("OpenAI rate limit exceeded. Please retry later.")
        except self.APIConnectionError as e:
            raise RuntimeError(f"OpenAI connection error: {str(e)}")
        except self.APIError as e:
            # Don't expose API key in error
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class AnthropicLLMProvider(LLMProvider):
    """Production Anthropic Claude provider using modern SDK."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None,
                 temperature: float = 0.2, max_tokens: int = 512, timeout: float = 30.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it in .env or environment before using Anthropic provider."
            )
        
        try:
            from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError
            self.client = Anthropic(api_key=self.api_key, timeout=timeout)
            self.APIError = APIError
            self.RateLimitError = RateLimitError
            self.APIConnectionError = APIConnectionError
        except ImportError:
            raise ImportError("anthropic>=0.7.0 required: pip install anthropic")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using Anthropic Claude."""
        max_tokens = max_tokens or self.max_tokens
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except self.RateLimitError:
            raise RuntimeError("Anthropic rate limit exceeded. Please retry later.")
        except self.APIConnectionError as e:
            raise RuntimeError(f"Anthropic connection error: {str(e)}")
        except self.APIError as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")


class GroqLLMProvider(LLMProvider):
    """Production Groq LLM provider using OpenAI-compatible API."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b", api_key: Optional[str] = None,
                 temperature: float = 0.2, max_tokens: int = 512, timeout: float = 30.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Set it in .env or environment before using Groq provider."
            )
        
        try:
            from openai import OpenAI, APIError, RateLimitError, APIConnectionError
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=timeout
            )
            self.APIError = APIError
            self.RateLimitError = RateLimitError
            self.APIConnectionError = APIConnectionError
        except ImportError:
            raise ImportError("openai>=1.0.0 required: pip install openai")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using Groq API."""
        max_tokens = max_tokens or self.max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except self.RateLimitError:
            raise RuntimeError("Groq rate limit exceeded. Please retry later.")
        except self.APIConnectionError as e:
            raise RuntimeError(f"Groq connection error: {str(e)}")
        except self.APIError as e:
            raise RuntimeError(f"Groq API error: {str(e)}")


def get_llm_provider(provider: Optional[str] = None, model: Optional[str] = None,
                     api_key: Optional[str] = None, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> LLMProvider:
    """
    Factory function to get LLM provider from environment or explicit parameters.
    
    Priority:
    1. Explicit parameters
    2. Environment variables: LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
    3. No fallback to mock if real provider is selected but credentials missing
    
    Args:
        provider: Provider name (openai, anthropic, groq, mock)
        model: Model name/identifier
        api_key: API key (if needed)
        temperature: Generation temperature
        max_tokens: Max output tokens
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If real provider selected but API key missing
    """
    provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
    
    # Load config from environment
    temperature = temperature or float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    max_tokens = max_tokens or int(os.environ.get("LLM_MAX_TOKENS", "512"))
    
    if provider == "openai":
        model = model or os.environ.get("LLM_MODEL", "gpt-4")
        return OpenAILLMProvider(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "anthropic":
        model = model or os.environ.get("LLM_MODEL", "claude-3-sonnet-20240229")
        return AnthropicLLMProvider(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "groq":
        model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
        return GroqLLMProvider(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "mock":
        model = model or os.environ.get("LLM_MODEL", "mock-model")
        return MockLLMProvider(model_name=model)
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai', 'anthropic', 'groq', or 'mock'.")
