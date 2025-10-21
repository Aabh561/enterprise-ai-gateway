# Working.md

This file provides guidance ( when working with code in this repository).

## Development Commands

### Local Development Setup
```bash
# Create virtual environment  
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env to add API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
```

### Running the Application
```bash
# Development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# With Docker Compose (recommended)
docker-compose up -d

# Production deployment
docker build --target production -t enterprise-ai-gateway:latest .
docker run -p 8000:8000 -e ENVIRONMENT=production enterprise-ai-gateway:latest
```

### Testing & Quality Assurance
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=services --cov-report=html

# Run specific test categories
pytest tests/unit/        # Unit tests
pytest tests/integration/ # Integration tests
pytest tests/e2e/         # End-to-end tests

# Code formatting and linting
black app/ services/ tests/
isort app/ services/ tests/
flake8 app/ services/ tests/
mypy app/ services/
```

### Service Management
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/health/ready

# Stop services
docker-compose down
```

## Architecture Overview

### Core Design Pattern
The system follows a **layered architecture** with strict separation of concerns:

- **API Layer** (`app/`): FastAPI application with middleware, routing, and request/response handling
- **Service Layer** (`services/`): Business logic and external integrations  
- **Configuration Layer**: Environment-specific YAML configs with Pydantic validation
- **Infrastructure Layer**: Docker services for databases, caching, and monitoring

### Key Architectural Components

#### 1. FastAPI Application Structure (`app/main.py`)
- **Lifespan management**: Async startup/shutdown for service initialization
- **Middleware stack**: Security, logging, CORS, metrics collection (Prometheus)
- **Exception handling**: Centralized error handling with structured logging
- **Health endpoints**: Kubernetes-ready health checks and metrics endpoint

#### 2. Configuration System (`app/config.py`)
- **Hierarchical config**: Environment-specific YAML files override base settings
- **Pydantic validation**: Type-safe configuration with automatic validation
- **Environment resolution**: Loads `configs/{environment}.yaml` based on `ENVIRONMENT` var
- **Secrets management**: Environment variable substitution for sensitive data

#### 3. Service Architecture Pattern
Services follow a consistent initialization pattern:
- Async `initialize()` method for startup tasks
- `health_check()` method for monitoring
- `close()` method for cleanup
- Services are dependency-injected via FastAPI's app state

#### 4. Multi-Provider LLM Integration
- **Provider abstraction**: Unified interface for OpenAI, Anthropic, Ollama
- **Model routing**: Dynamic provider/model selection per request
- **Failover support**: Built-in retry and fallback mechanisms
- **Local LLM support**: Ollama integration for on-premises deployment

#### 5. Vector Database & RAG Pipeline
- **Vector store**: Weaviate integration for document embeddings
- **Document processing**: Unstructured.io for PDF, DOCX, HTML parsing
- **Embedding models**: Sentence-Transformers with configurable models
- **Retrieval**: Context-aware document retrieval for LLM augmentation

#### 6. Security & Compliance Layer
- **PII Protection**: Microsoft Presidio integration for automatic detection/redaction
- **Authentication**: API key and JWT-based authentication
- **Audit logging**: Immutable audit trails with request tracing
- **Rate limiting**: Redis-backed rate limiting per user/endpoint

### Docker Services Architecture
The application runs in a multi-container setup:

- **App**: Main FastAPI application
- **PostgreSQL**: Primary data persistence
- **Redis**: Caching and session management
- **Weaviate**: Vector database for embeddings
- **Ollama**: Local LLM server
- **Prometheus/Grafana**: Metrics and monitoring
- **Presidio**: PII detection/anonymization services
- **Nginx**: Load balancing (production profile)

### Environment Configuration
Three environment configurations:
- **Development** (`configs/dev.yaml`): Debug enabled, SQLite database
- **Production** (`configs/prod.yaml`): Optimized for deployment
- **Testing**: Isolated test environment

### Plugin System
- **Directory-based**: Plugins loaded from `./plugins/` directory
- **Sandbox support**: Optional resource-constrained execution
- **Built-in plugins**: SQL agent, financial API agent
- **Plugin interface**: Standardized `BasePlugin` class for extensions

### Monitoring & Observability
- **Structured logging**: JSON-based logs with request correlation
- **Prometheus metrics**: HTTP requests, LLM usage, cache performance
- **Health checks**: Liveness and readiness probes
- **Distributed tracing**: Request ID propagation across services

## Project-Specific Guidelines

### Configuration Management
- Always use environment-specific YAML files in `configs/`
- Use environment variable substitution for secrets (e.g., `${API_KEY}`)
- Test configuration loading with `get_config_for_environment(env_name)`

### Service Development
- Follow the established service pattern (initialize/health_check/close)
- Add services to `app.state` during lifespan startup
- Use structured logging with request IDs for traceability

### API Development  
- Use Pydantic models in `app/models.py` for request/response validation
- Include proper OpenAPI documentation with examples
- Follow RESTful conventions for endpoint naming

### Database Operations
- Use async SQLAlchemy for database operations
- Implement proper connection pooling via configuration
- Include database migrations in deployment processes

### Testing Strategy
- Write unit tests for individual components
- Integration tests for service interactions
- End-to-end tests for complete workflows
- Use pytest fixtures for test data setup

### Deployment Considerations
- Use multi-stage Dockerfiles (development/production targets)
- Configure environment-specific docker-compose overrides
- Implement proper secret management for production
- Monitor resource usage and adjust container limits accordingly
