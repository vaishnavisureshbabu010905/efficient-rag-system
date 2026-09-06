# Milestone 6: Production API Deployment

## Overview

M6 exposes the complete M1-M5 RAG pipeline through a production-ready FastAPI service.

**Architecture**: M1-M5 pipeline → FastAPI endpoints → REST API

## Features

### Endpoints

**GET /health**
- Simple health check
- Returns: `{"status": "healthy", "version": "1.0.0"}`
- No authentication required

**POST /query**
- Execute RAG query
- Request: `{"query": "...", "top_k": 5}`
- Response: Answer, sources, metadata, latency
- Validates query length and top_k range
- Returns 400 for invalid input, 500 for server errors

### API Components

**FastAPI Application** (`src/efficient_rag/api/main.py`)
- RESTful endpoints
- Automatic OpenAPI/Swagger docs at `/docs` and `/redoc`
- Error handling and exception handlers
- CORS middleware (configurable)

**Schemas** (`src/efficient_rag/api/schemas.py`)
- `QueryRequest`: Query input with validation
- `QueryResponse`: Full response with metadata
- `HealthResponse`: Health check response
- `ErrorResponse`: Error messages
- All use Pydantic for validation

**Dependencies** (`src/efficient_rag/api/dependencies.py`)
- `RAGPipeline`: Encapsulates M1-M5 pipeline initialization
- `get_rag_pipeline()`: Singleton factory with caching
- Lazy initialization on first request

## Configuration

### Environment Variables

```bash
# LLM provider (mock, openai, anthropic)
LLM_PROVIDER=mock

# LLM model name
LLM_MODEL=mock-model

# API key (if needed)
LLM_API_KEY=your_key

# API server
API_HOST=0.0.0.0
API_PORT=8000

# Document path
DOCUMENTS_PATH=data/sample_documents

# Context assembly
MAX_CONTEXT_TOKENS=2000

# CORS
CORS_ORIGINS=*
```

See `.env.example` for all options.

## Local Development

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Run API

```bash
uvicorn src.efficient_rag.api.main:app --reload
```

Navigate to http://localhost:8000/docs for interactive API documentation.

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 3
  }'
```

## Docker

### Build Image

```bash
docker build -t efficient-rag-api:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e LLM_PROVIDER=mock \
  -e DOCUMENTS_PATH=/app/data/sample_documents \
  -v $(pwd)/data:/app/data \
  efficient-rag-api:latest
```

Access at http://localhost:8000/docs

### With Environment File

```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  efficient-rag-api:latest
```

## Testing

Run API tests:

```bash
pytest tests/test_api.py -v
```

All tests (M1-M6):

```bash
pytest tests/ -v
```

## Request/Response Examples

### Health Check

**Request:**
```
GET /health
```

**Response (200):**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Query

**Request:**
```json
POST /query
{
  "query": "What is machine learning?",
  "top_k": 3
}
```

**Response (200):**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a type of AI...",
  "sources": ["doc0_0", "doc0_1"],
  "has_sufficient_context": true,
  "num_chunks_retrieved": 2,
  "retrieval_metadata": [
    {
      "chunk_id": "doc0_0",
      "fusion_score": 0.95,
      "rank": 1
    }
  ],
  "latency_ms": 125.5
}
```

### Validation Error

**Request:**
```json
POST /query
{
  "query": "",
  "top_k": 100
}
```

**Response (422):**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "query"],
      "msg": "Query cannot be empty or whitespace only"
    }
  ]
}
```

## Error Handling

| Status | Scenario |
|--------|----------|
| 200 | Successful query |
| 400 | Invalid input (empty query, etc.) |
| 404 | Endpoint not found |
| 405 | Wrong HTTP method |
| 422 | Validation error |
| 500 | Server error (LLM failure, etc.) |

Error responses include:
- `error`: Human-readable message
- `status_code`: HTTP status
- `detail`: Optional additional info

Secrets and stack traces are NOT exposed to clients.

## Pipeline Integration

The API uses existing M1-M5 components without duplication:

```
Request → QueryRequest Validation
  ↓
RAGPipeline.query() → M1-M5 Pipeline
  ├─ M1: Load chunks from documents
  ├─ M2: Dense embedding + FAISS index
  ├─ M3: Binary quantization + Hamming index
  ├─ M4: Hybrid retrieval (RRF/weighted avg)
  └─ M5: LLM generation with grounding
  ↓
Response → QueryResponse serialization
```

## Design Notes

**Why singleton pipeline?**
- Load documents once on startup
- Reuse embedders and indexes across requests
- Efficient resource usage

**Why lazy initialization?**
- Pipeline loads on first request
- Faster startup time
- Fails gracefully if documents missing

**Why Pydantic schemas?**
- Automatic validation
- Type safety
- OpenAPI documentation

**Why CORS middleware?**
- Allows future frontend integration
- Configurable origins

## Monitoring & Production

For production deployments:
- Use production ASGI server (Gunicorn, uWSGI, etc.)
- Add authentication/authorization
- Implement request logging
- Monitor LLM API usage/costs
- Add request rate limiting
- Use reverse proxy (Nginx)
- Enable HTTPS
- Add caching layer

Example with Gunicorn:
```bash
gunicorn src.efficient_rag.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Limitations

- No authentication/authorization
- No request rate limiting
- No response caching
- Documents loaded into memory
- Single-threaded for M1-M5 (but FastAPI handles concurrency at request level)
- Mock LLM for demo (real LLMs need API keys)

## Next Steps (M7+)

- Authentication and API keys
- Request rate limiting
- Response caching
- Advanced monitoring
- Load balancing
- Cloud deployment (AWS/GCP/Azure)

---

**Status**: M6 Complete | API fully functional with M1-M5 integration
