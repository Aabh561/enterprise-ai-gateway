# Enterprise AI Gateway 🚀

**The Central Nervous System for Enterprise AI**

A secure, intelligent, and observable platform that acts as the single point of entry for all Large Language Model (LLM) interactions within a company. Built to solve the enterprise AI crisis by eliminating model lock-in, enforcing security and compliance, providing verifiable context-aware answers, and offering complete operational control.

## 🎯 Mission

Our mission is to build the **Enterprise AI Gateway**: a comprehensive platform that:

- **Eliminates Model Lock-in**: Support multiple LLM providers seamlessly
- **Enforces Security & Compliance**: Advanced PII protection and audit trails
- **Provides Verifiable Answers**: RAG with source citations and context
- **Offers Complete Control**: Full observability, monitoring, and governance

## ⚙️ Core Technology Stack

This stack is chosen for performance, scalability, and prevalence in top tech companies:

### Backend & Framework
- **FastAPI**: High-performance async API framework with automatic OpenAPI docs
- **Pydantic V2**: Strict data validation and settings management
- **Python 3.11+**: Modern Python with enhanced performance

### Data & Configuration
- **YAML Configuration**: Environment-specific profiles (dev, prod)
- **Structured Logging**: JSON-based logs with request tracing
- **Environment Variables**: Secure configuration management

### LLM Integration
- **Multi-Provider Support**: OpenAI, Anthropic, Ollama (local models)
- **LangChain Integration**: High-level framework for advanced workflows
- **Local LLM Server**: Ollama for running open-source models locally

### Data Processing & Storage
- **Vector Database**: Weaviate/Milvus for production-grade vector search
- **Document Processing**: Unstructured.io for parsing PDF, DOCX, HTML
- **Embedding Models**: Sentence-Transformers with state-of-the-art models
- **PostgreSQL**: Primary data storage with async operations
- **Redis**: High-performance caching and session management

### Security & Compliance
- **PII Protection**: Microsoft Presidio for detection and redaction
- **API Key Management**: Secure authentication and authorization
- **Audit Logging**: Immutable trail for compliance requirements
- **Rate Limiting**: Protect against abuse and manage costs

### Infrastructure & Deployment
- **Docker & Docker Compose**: Containerized deployment
- **Kubernetes**: Production orchestration with Kustomize
- **Terraform**: Infrastructure as Code for cloud resources
- **GitHub Actions**: Automated CI/CD pipeline

### Observability & Monitoring
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Beautiful dashboards and visualization
- **Structured Logging**: Request tracing and debugging
- **Health Checks**: Kubernetes-ready health endpoints

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Security Layer  │────│ Service Layer   │
│   (FastAPI)     │    │  (Auth, Rate     │    │ (Chat, RAG,     │
│                 │    │   Limiting)      │    │  Reasoning)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Vector Database │    │   LLM Providers  │    │ Plugin System   │
│ (Weaviate)      │    │ (OpenAI, Claude, │    │ (SQL, Finance,  │
│                 │    │  Ollama)         │    │  Custom Tools)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **Git**

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/enterprise-ai-gateway.git
cd enterprise-ai-gateway

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (add your API keys)
# Required: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### 3. Start Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Wait for services to be ready (check logs)
docker-compose logs -f app
```

### 4. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## 📚 API Usage Examples

### Authentication

All API calls require an API key:

```bash
curl -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/v1/status
```

### Chat Generation

```bash
curl -X POST "http://localhost:8000/api/v1/chat/generate" \
     -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Explain quantum computing in simple terms",
       "provider": "openai",
       "model": "gpt-4",
       "temperature": 0.7,
       "use_rag": true
     }'
```

### Streaming Chat

```bash
curl -X POST "http://localhost:8000/api/v1/chat/stream" \
     -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Write a Python function to calculate fibonacci numbers",
       "stream": true
     }'
```

### Document Ingestion (RAG)

```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest" \
     -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": [
         {
           "content": "Your document content here...",
           "metadata": {"title": "Company Policies", "type": "policy"}
         }
       ],
       "collection_name": "company_docs"
     }'
```

## 🔧 Development Setup

### Local Development

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Start in development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v --cov=app --cov=services

# Code quality checks
black app/ services/ tests/
isort app/ services/ tests/
flake8 app/ services/ tests/
mypy app/ services/
```

### Environment Configuration

The application supports multiple environments:

- **Development**: `configs/dev.yaml`
- **Production**: `configs/prod.yaml`
- **Testing**: `configs/test.yaml`

Set the `ENVIRONMENT` variable to switch between configurations.

## 🔒 Security Features

### PII Protection

Automatic detection and redaction of sensitive information:

```python
# PII is automatically detected and masked
request: "My SSN is 123-45-6789"
processed: "My SSN is [PERSON_1]"
response: "I can help with [PERSON_1]'s request"
final: "I can help with your request"  # Unmasked for user
```

### API Security

- **API Key Authentication**: Secure access control
- **Rate Limiting**: Prevent abuse and manage costs
- **Request Size Limits**: Prevent DoS attacks
- **HTTPS Enforcement**: Secure communication (production)
- **CORS Configuration**: Cross-origin request management

### Audit Logging

Every request generates an immutable audit trail:

```json
{
  "request_id": "abc123",
  "timestamp": "2024-01-01T12:00:00Z",
  "user_id": "user123",
  "action": "chat_request",
  "input_tokens": 50,
  "output_tokens": 200,
  "model": "gpt-4",
  "cost_usd": 0.012,
  "pii_detected": false,
  "plugins_used": ["sql_agent"]
}
```

## 🔌 Plugin System

### Built-in Plugins

1. **SQL Database Agent**: Natural language database queries
2. **Financial API Agent**: Real-time financial data integration
3. **Document Search**: RAG-powered document retrieval

### Custom Plugin Development

```python
from plugins.base import BasePlugin

class CustomPlugin(BasePlugin):
    def __init__(self):
        super().__init__(
            name="custom_plugin",
            description="Custom functionality",
            version="1.0.0"
        )
    
    async def execute(self, action: str, parameters: dict) -> dict:
        # Your custom logic here
        return {"result": "success"}
```

## 📊 Monitoring & Observability

### Metrics Dashboard

Access Grafana dashboard at `http://localhost:3000`:

- **Request Metrics**: Rate, latency, errors
- **LLM Metrics**: Token usage, costs, model performance
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: User activity, plugin usage

### Key Metrics

```prometheus
# HTTP request metrics
http_requests_total{method="POST", endpoint="/api/v1/chat/generate", status="200"}

# LLM usage metrics
llm_requests_total{provider="openai", model="gpt-4", status="success"}
llm_tokens_total{type="input", provider="openai"}
llm_cost_usd_total{provider="openai", model="gpt-4"}

# Cache performance
cache_hits_total{type="response"}
cache_miss_total{type="response"}
```

## 🚀 Deployment

### Docker Production Deployment

```bash
# Build production image
docker build --target production -t enterprise-ai-gateway:latest .

# Run in production mode
docker run -p 8000:8000 \
           -e ENVIRONMENT=production \
           -e DATABASE_URL=your-db-url \
           enterprise-ai-gateway:latest
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f infrastructure/k8s/

# Check deployment status
kubectl get pods -l app=enterprise-ai-gateway
```

### Infrastructure as Code

```bash
# Deploy with Terraform
cd infrastructure/terraform/
terraform init
terraform plan
terraform apply
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=services --cov-report=html

# Run specific test categories
pytest tests/unit/        # Unit tests
pytest tests/integration/ # Integration tests
pytest tests/e2e/         # End-to-end tests
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Service integration testing  
- **End-to-End Tests**: Full workflow testing
- **Performance Tests**: Load and stress testing

## 📈 Performance & Scaling

### Performance Characteristics

- **Response Time**: < 100ms for cached responses
- **Throughput**: 1000+ requests/second (single instance)
- **Concurrency**: Async handling for high concurrent loads
- **Caching**: Multi-layer caching (Redis, in-memory)

### Scaling Strategies

1. **Horizontal Scaling**: Multiple app instances behind load balancer
2. **Database Scaling**: Read replicas, connection pooling
3. **Cache Scaling**: Redis clustering, cache warming
4. **LLM Scaling**: Provider failover, request queuing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Community

- **Documentation**: [docs.enterprise-ai-gateway.com](https://docs.enterprise-ai-gateway.com)
- **Issues**: [GitHub Issues](https://github.com/enterprise-ai/enterprise-ai-gateway/issues)
- **Discussions**: [GitHub Discussions](https://github.com/enterprise-ai/enterprise-ai-gateway/discussions)
- **Slack**: [Join our community](https://enterprise-ai-gateway.slack.com)

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Core API framework
- ✅ Multi-provider LLM support
- ✅ Basic security features
- ✅ Docker deployment

### Version 1.1 (Next)
- 🔄 Advanced RAG pipeline
- 🔄 Plugin marketplace
- 🔄 Enhanced monitoring
- 🔄 Kubernetes operators

### Version 2.0 (Future)
- 🔮 AI-powered optimization
- 🔮 Multi-tenant architecture
- 🔮 Advanced compliance features
- 🔮 Enterprise SSO integration

---

**Built with ❤️ for Enterprise AI**

*Transforming how organizations interact with AI, one request at a time.*