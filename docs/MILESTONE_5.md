# Milestone 5: RAG Generation (Retrieval-Augmented Generation)

## Overview

M5 implements the final layer of the RAG system: using retrieved context to generate grounded answers.

**Pipeline**: M1 (chunks) → M2 (dense) + M3 (binary) → M4 (hybrid fusion) → **M5 (generation)**

## Architecture

```
Query
  ├─→ QueryProcessor (preprocessing)
  │
  ├─→ M4 HybridRetriever (context retrieval)
  │
  ├─→ Context Assembly (chunk → text)
  │
  ├─→ Prompt Building (template + context + query)
  │
  ├─→ LLMProvider (generate answer)
  │
  └─→ Response (answer + sources + metadata)
```

## Components

### LLMProvider (Abstraction)

Pluggable LLM backend supporting multiple providers via environment variables.

```python
from src.efficient_rag.milestone5.llm_provider import get_llm_provider

# Default (mock): no API calls, for testing
llm = get_llm_provider()  # MockLLMProvider

# OpenAI (requires OPENAI_API_KEY)
llm = get_llm_provider(provider="openai", model="gpt-3.5-turbo")

# Anthropic Claude (requires ANTHROPIC_API_KEY)
llm = get_llm_provider(provider="anthropic", model="claude-3-sonnet-20240229")
```

**Providers**:
- **MockLLMProvider**: No API calls (default for demo/testing)
- **OpenAILLMProvider**: GPT-3.5-turbo, GPT-4, etc. (requires `openai` package)
- **AnthropicLLMProvider**: Claude models (requires `anthropic` package)

**Configuration (environment variables)**:
```bash
export LLM_PROVIDER=openai          # or "anthropic", "mock"
export LLM_MODEL=gpt-3.5-turbo      # model name
export LLM_API_KEY=sk-...           # API key
```

### RAGGenerator

End-to-end RAG with retrieval, context assembly, and answer generation.

```python
from src.efficient_rag.milestone5.rag_generator import RAGGenerator

rag = RAGGenerator(
    hybrid_retriever=m4_hybrid,
    llm_provider=llm,
    max_context_tokens=2000,
    min_chunk_score=0.5,
    context_window=4096
)

# Generate grounded answer
result = rag.generate(query="What is machine learning?", top_k=5)

# Result structure:
# {
#   "query": original query,
#   "answer": generated answer,
#   "sources": [chunk_ids],
#   "retrieval_results": raw retrieval data,
#   "has_sufficient_context": bool,
#   "num_chunks_retrieved": int
# }
```

**Key Features**:
- **Context Assembly**: Assembles retrieved chunks into context, respecting token limits
- **Prompt Templates**: Clear, grounded prompts with context and query
- **Source Tracking**: Records which chunks were used (for attribution)
- **Insufficient Context Handling**: Detects when retrieval found no relevant chunks
- **Score Filtering**: Optional minimum score threshold for retrieved chunks

### QueryProcessor

Simple query preprocessing and expansion.

```python
from src.efficient_rag.milestone5.rag_generator import QueryProcessor

# Preprocess (normalize whitespace)
cleaned = QueryProcessor.preprocess("  What is AI?  ")  # "What is AI?"

# Expand (generate variants for multi-attempt retrieval)
variants = QueryProcessor.expand_query("machine learning")
```

## Full Pipeline Example

```python
from efficient_rag.milestone1.document_processor import DocumentProcessor
from efficient_rag.milestone2.embedding import DenseEmbedder
from efficient_rag.milestone2.retriever import DenseRetriever
from efficient_rag.milestone3.quantizer import BinaryQuantizer
from efficient_rag.milestone3.binary_retriever import BinaryRetriever
from efficient_rag.milestone4.hybrid_retriever import HybridRetriever
from efficient_rag.milestone5.llm_provider import get_llm_provider
from efficient_rag.milestone5.rag_generator import RAGGenerator

# M1: Load chunks
processor = DocumentProcessor()
chunks = processor.process_pipeline('data/documents')

# M2: Dense index
embedder = DenseEmbedder()
dense_retriever = DenseRetriever(embedder)
dense_retriever.build_index(chunks)

# M3: Binary index
quantizer = BinaryQuantizer(dim=384)
binary_embeddings = quantizer.quantize(
    embedder.encode_chunks([c["text"] for c in chunks])
)
binary_retriever = BinaryRetriever(quantizer)
binary_retriever.build_index(chunks, binary_embeddings)

# M4: Hybrid retrieval
hybrid = HybridRetriever(
    dense_retriever, binary_retriever, quantizer, embedder,
    fusion_strategy="rrf"
)

# M5: RAG generation
llm = get_llm_provider()
rag = RAGGenerator(hybrid, llm)

# Generate
result = rag.generate("What is machine learning?", top_k=5)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## Configuration

### Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=openai|anthropic|mock     # Default: mock
LLM_MODEL=model_name                    # Default depends on provider
LLM_API_KEY=your_key                    # For OpenAI/Anthropic

# Context
MAX_CONTEXT_TOKENS=2000                 # Max tokens in context
MIN_CHUNK_SCORE=0.5                     # Min fusion score threshold
CONTEXT_WINDOW=4096                     # Total prompt context window
```

### Runtime Configuration

```python
rag = RAGGenerator(
    hybrid_retriever,
    llm_provider,
    max_context_tokens=2000,     # Adjust for your use case
    min_chunk_score=0.6,         # Filter low-quality results
    context_window=4096          # Model's context size
)
```

## Prompt Design

M5 uses clear, grounded prompts:

**With Context**:
```
You are a helpful assistant answering questions based on provided context.

Context:
[retrieved chunk 1]
[retrieved chunk 2]
...

Question: {user_query}

Answer: Based on the context above, provide a clear and concise answer...
```

**Without Context**:
```
You are a helpful assistant. The user asked:

{user_query}

Unfortunately, no relevant information was found in the knowledge base...
```

## Testing

17 tests covering:
- LLM providers (mock, factory)
- Query processing
- RAG generation
- Context assembly
- Source tracking
- Insufficient context handling
- Integration pipeline

## Design Notes

**Why abstracted LLM provider?**
- Support multiple backends (OpenAI, Anthropic, local models)
- Easy to add new providers
- Environment-based configuration
- Testing with mock provider

**Why context assembly?**
- Respects context window limits
- Avoids token overflow
- Maintains chunk ordering
- Graceful degradation

**Why source tracking?**
- Attribution and transparency
- Fact-checking and verification
- Audit trail
- User trust

**No streaming in M5?**
- Simplified for this milestone
- Can be added for production (M6+)
- Generator method provided as placeholder

## Limitations

- Mock provider for demo (no real LLM calls)
- Basic prompt templates (can be more sophisticated)
- No streaming (for speed)
- Context assembly is simple (no reranking)

## Next Steps (M6+)

- M6: FastAPI deployment
- M7: Advanced prompting (few-shot, chain-of-thought)
- M8: Streaming generation
- M9: Response caching

## External Dependencies

**Optional** (for production use):
- `openai>=0.27.0` (for OpenAI provider)
- `anthropic>=0.7.0` (for Anthropic provider)

**Required** (already in M1-M4):
- `sentence-transformers`
- `faiss-cpu`
- `pdfplumber`
- `numpy`

---

**Status**: M5 Complete | Tests: 74/74 passing (M1-M5)
