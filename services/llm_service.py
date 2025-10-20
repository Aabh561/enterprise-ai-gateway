"""
Advanced LLM Service for Enterprise AI Gateway

Provides intelligent routing, cost optimization, provider fallback,
and comprehensive monitoring for multiple LLM providers.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import structlog
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from langchain_community.llms import Ollama
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(__name__)

# Metrics
LLM_REQUEST_COUNTER = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['provider', 'model', 'status', 'user_id']
)

LLM_LATENCY_HISTOGRAM = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['provider', 'model']
)

LLM_TOKEN_COUNTER = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['provider', 'model', 'type']  # type: input/output
)

LLM_COST_COUNTER = Counter(
    'llm_cost_usd_total',
    'Total cost in USD',
    ['provider', 'model']
)

ACTIVE_REQUESTS_GAUGE = Gauge(
    'llm_active_requests',
    'Active LLM requests',
    ['provider']
)


class LLMProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"


@dataclass
class LLMRequest:
    """LLM request data structure."""
    message: str
    model: str = ""
    provider: Optional[LLMProviderType] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    system_prompt: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM response data structure."""
    content: str
    provider: LLMProviderType
    model: str
    usage: Dict[str, int]
    cost_usd: float
    latency_ms: float
    cached: bool = False
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Provider configuration."""
    provider_type: LLMProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    models: List[str] = field(default_factory=list)
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    rate_limit_rpm: int = 1000
    priority: int = 1  # Higher priority = preferred provider
    health_check_interval: int = 60
    enabled: bool = True


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None
        self._healthy = True
        self._last_health_check = 0
        
    @property
    def provider_type(self) -> LLMProviderType:
        return self.config.provider_type
        
    @property
    def is_healthy(self) -> bool:
        # Check if health check is needed
        current_time = time.time()
        if (current_time - self._last_health_check) > self.config.health_check_interval:
            asyncio.create_task(self.health_check())
        return self._healthy
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health."""
        pass
    
    async def close(self) -> None:
        """Clean up provider resources."""
        if hasattr(self.client, 'close'):
            await self.client.close()


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    async def initialize(self) -> None:
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        
        # Prepare messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        # Add conversation history
        messages.extend(request.conversation_history)
        messages.append({"role": "user", "content": request.message})
        
        try:
            response = await self.client.chat.completions.create(
                model=request.model or "gpt-4",
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream
            )
            
            content = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Calculate cost
            cost = (
                usage["input_tokens"] / 1000 * self.config.cost_per_1k_input_tokens +
                usage["output_tokens"] / 1000 * self.config.cost_per_1k_output_tokens
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=content,
                provider=self.provider_type,
                model=request.model or "gpt-4",
                usage=usage,
                cost_usd=cost,
                latency_ms=latency_ms,
                request_id=request.request_id
            )
            
        except Exception as e:
            logger.error("OpenAI request failed", error=str(e), request_id=request.request_id)
            raise
    
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        messages.extend(request.conversation_history)
        messages.append({"role": "user", "content": request.message})
        
        try:
            response = await self.client.chat.completions.create(
                model=request.model or "gpt-4",
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("OpenAI streaming failed", error=str(e), request_id=request.request_id)
            raise
    
    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            self._healthy = True
            self._last_health_check = time.time()
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            self._healthy = False
            self._last_health_check = time.time()
            return False


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    async def initialize(self) -> None:
        self.client = AsyncAnthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        
        # Format prompt for Claude
        prompt = request.message
        if request.system_prompt:
            prompt = f"{request.system_prompt}\\n\\nHuman: {prompt}\\n\\nAssistant:"
        else:
            prompt = f"Human: {prompt}\\n\\nAssistant:"
        
        try:
            response = await self.client.completions.create(
                model=request.model or "claude-2",
                prompt=prompt,
                temperature=request.temperature,
                max_tokens_to_sample=request.max_tokens
            )
            
            content = response.completion
            usage = {
                "input_tokens": len(prompt.split()) * 1.3,  # Rough estimate
                "output_tokens": len(content.split()) * 1.3,
                "total_tokens": len(prompt.split()) * 1.3 + len(content.split()) * 1.3
            }
            
            cost = (
                usage["input_tokens"] / 1000 * self.config.cost_per_1k_input_tokens +
                usage["output_tokens"] / 1000 * self.config.cost_per_1k_output_tokens
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=content,
                provider=self.provider_type,
                model=request.model or "claude-2",
                usage=usage,
                cost_usd=cost,
                latency_ms=latency_ms,
                request_id=request.request_id
            )
            
        except Exception as e:
            logger.error("Anthropic request failed", error=str(e), request_id=request.request_id)
            raise
    
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        # Anthropic streaming implementation
        prompt = f"Human: {request.message}\\n\\nAssistant:"
        if request.system_prompt:
            prompt = f"{request.system_prompt}\\n\\n{prompt}"
        
        try:
            response = await self.client.completions.create(
                model=request.model or "claude-2",
                prompt=prompt,
                temperature=request.temperature,
                max_tokens_to_sample=request.max_tokens,
                stream=True
            )
            
            async for chunk in response:
                if chunk.completion:
                    yield chunk.completion
                    
        except Exception as e:
            logger.error("Anthropic streaming failed", error=str(e), request_id=request.request_id)
            raise
    
    async def health_check(self) -> bool:
        try:
            # Simple test completion
            await self.client.completions.create(
                model="claude-instant-1",
                prompt="Human: Test\\n\\nAssistant:",
                max_tokens_to_sample=5
            )
            self._healthy = True
            self._last_health_check = time.time()
            return True
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            self._healthy = False
            self._last_health_check = time.time()
            return False


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    async def initialize(self) -> None:
        self.client = Ollama(
            base_url=self.config.base_url or "http://localhost:11434",
            timeout=self.config.timeout
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Format prompt
            prompt = request.message
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\\n\\nUser: {prompt}\\n\\nAssistant:"
            
            response = await self.client.agenerate([prompt])
            content = response.generations[0][0].text
            
            # Estimate usage (Ollama doesn't provide exact token counts)
            usage = {
                "input_tokens": len(prompt.split()) * 1.3,
                "output_tokens": len(content.split()) * 1.3,
                "total_tokens": len(prompt.split()) * 1.3 + len(content.split()) * 1.3
            }
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=content,
                provider=self.provider_type,
                model=request.model or "llama3",
                usage=usage,
                cost_usd=0.0,  # Local model, no cost
                latency_ms=latency_ms,
                request_id=request.request_id
            )
            
        except Exception as e:
            logger.error("Ollama request failed", error=str(e), request_id=request.request_id)
            raise
    
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        # Ollama streaming would require direct HTTP calls
        prompt = request.message
        if request.system_prompt:
            prompt = f"System: {request.system_prompt}\\n\\nUser: {prompt}\\n\\nAssistant:"
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.config.base_url}/api/generate",
                    json={
                        "model": request.model or "llama3",
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                logger.error("Ollama streaming failed", error=str(e), request_id=request.request_id)
                raise
    
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.base_url}/api/tags")
                if response.status_code == 200:
                    self._healthy = True
                    self._last_health_check = time.time()
                    return True
                else:
                    self._healthy = False
                    return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            self._healthy = False
            self._last_health_check = time.time()
            return False


class LLMLoadBalancer:
    """Intelligent load balancer for LLM providers."""
    
    def __init__(self):
        self.providers: Dict[LLMProviderType, List[LLMProvider]] = {}
        self.request_counts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}
        
    def add_provider(self, provider: LLMProvider) -> None:
        """Add a provider to the load balancer."""
        if provider.provider_type not in self.providers:
            self.providers[provider.provider_type] = []
        self.providers[provider.provider_type].append(provider)
        
    def select_provider(
        self, 
        provider_type: Optional[LLMProviderType] = None,
        model: Optional[str] = None
    ) -> Optional[LLMProvider]:
        """Select the best provider based on health, load, and priority."""
        
        # Filter by provider type if specified
        candidates = []
        if provider_type:
            candidates = self.providers.get(provider_type, [])
        else:
            # Consider all providers
            for providers in self.providers.values():
                candidates.extend(providers)
        
        # Filter by model availability
        if model:
            candidates = [
                p for p in candidates 
                if not p.config.models or model in p.config.models
            ]
        
        # Filter by health
        healthy_candidates = [p for p in candidates if p.is_healthy and p.config.enabled]
        
        if not healthy_candidates:
            return None
        
        # Sort by priority (higher is better) and then by load (lower is better)
        def score_provider(provider: LLMProvider) -> tuple:
            provider_key = f"{provider.provider_type}_{id(provider)}"
            failures = self.failure_counts.get(provider_key, 0)
            requests = self.request_counts.get(provider_key, 0)
            
            # Higher priority, lower failures, lower requests = better
            return (-provider.config.priority, failures, requests)
        
        healthy_candidates.sort(key=score_provider)
        return healthy_candidates[0]
    
    def record_request(self, provider: LLMProvider, success: bool = True) -> None:
        """Record a request for load balancing."""
        provider_key = f"{provider.provider_type}_{id(provider)}"
        
        self.request_counts[provider_key] = self.request_counts.get(provider_key, 0) + 1
        
        if not success:
            self.failure_counts[provider_key] = self.failure_counts.get(provider_key, 0) + 1


class LLMService:
    """Advanced LLM service with intelligent routing and optimization."""
    
    def __init__(self, cache_service=None, security_service=None):
        self.providers: Dict[str, LLMProvider] = {}
        self.load_balancer = LLMLoadBalancer()
        self.cache_service = cache_service
        self.security_service = security_service
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the LLM service."""
        if self._initialized:
            return
            
        # Initialize all providers
        for provider in self.providers.values():
            try:
                await provider.initialize()
                self.load_balancer.add_provider(provider)
                logger.info(f"Initialized provider: {provider.provider_type}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider.provider_type}: {e}")
        
        self._initialized = True
        logger.info("LLM service initialized")
    
    def add_provider(self, provider: LLMProvider) -> None:
        """Add a new provider to the service."""
        provider_key = f"{provider.provider_type}_{id(provider)}"
        self.providers[provider_key] = provider
        
        if self._initialized:
            # Initialize immediately if service is already running
            asyncio.create_task(provider.initialize())
            self.load_balancer.add_provider(provider)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the best available provider."""
        
        # Security check
        if self.security_service:
            request.message = await self.security_service.sanitize_input(request.message)
        
        # Check cache
        cache_key = None
        if self.cache_service and not request.stream:
            cache_key = self._generate_cache_key(request)
            cached_response = await self.cache_service.get(cache_key)
            if cached_response:
                logger.info("Cache hit", request_id=request.request_id)
                cached_response.cached = True
                return cached_response
        
        # Select provider
        provider = self.load_balancer.select_provider(request.provider, request.model)
        if not provider:
            raise Exception("No healthy providers available")
        
        # Generate response
        start_time = time.time()
        success = True
        
        with ACTIVE_REQUESTS_GAUGE.labels(provider=provider.provider_type).track():
            try:
                response = await provider.generate(request)
                
                # Record metrics
                LLM_REQUEST_COUNTER.labels(
                    provider=provider.provider_type,
                    model=response.model,
                    status="success",
                    user_id=request.user_id or "anonymous"
                ).inc()
                
                LLM_LATENCY_HISTOGRAM.labels(
                    provider=provider.provider_type,
                    model=response.model
                ).observe(response.latency_ms / 1000)
                
                LLM_TOKEN_COUNTER.labels(
                    provider=provider.provider_type,
                    model=response.model,
                    type="input"
                ).inc(response.usage["input_tokens"])
                
                LLM_TOKEN_COUNTER.labels(
                    provider=provider.provider_type,
                    model=response.model,
                    type="output"
                ).inc(response.usage["output_tokens"])
                
                LLM_COST_COUNTER.labels(
                    provider=provider.provider_type,
                    model=response.model
                ).inc(response.cost_usd)
                
                # Cache response
                if self.cache_service and cache_key:
                    await self.cache_service.set(cache_key, response)
                
                # Security check on response
                if self.security_service:
                    response.content = await self.security_service.sanitize_output(response.content)
                
                return response
                
            except Exception as e:
                success = False
                LLM_REQUEST_COUNTER.labels(
                    provider=provider.provider_type,
                    model=request.model or "unknown",
                    status="error",
                    user_id=request.user_id or "anonymous"
                ).inc()
                
                logger.error(
                    "LLM generation failed",
                    provider=provider.provider_type,
                    error=str(e),
                    request_id=request.request_id
                )
                raise
            finally:
                self.load_balancer.record_request(provider, success)
    
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate a streaming response."""
        
        # Security check
        if self.security_service:
            request.message = await self.security_service.sanitize_input(request.message)
        
        # Select provider
        provider = self.load_balancer.select_provider(request.provider, request.model)
        if not provider:
            raise Exception("No healthy providers available for streaming")
        
        success = True
        
        with ACTIVE_REQUESTS_GAUGE.labels(provider=provider.provider_type).track():
            try:
                async for chunk in provider.stream_generate(request):
                    # Security check on each chunk
                    if self.security_service:
                        chunk = await self.security_service.sanitize_output(chunk)
                    yield chunk
                    
                # Record success metrics
                LLM_REQUEST_COUNTER.labels(
                    provider=provider.provider_type,
                    model=request.model or "unknown",
                    status="success_stream",
                    user_id=request.user_id or "anonymous"
                ).inc()
                
            except Exception as e:
                success = False
                LLM_REQUEST_COUNTER.labels(
                    provider=provider.provider_type,
                    model=request.model or "unknown",
                    status="error_stream",
                    user_id=request.user_id or "anonymous"
                ).inc()
                
                logger.error(
                    "LLM streaming failed",
                    provider=provider.provider_type,
                    error=str(e),
                    request_id=request.request_id
                )
                raise
            finally:
                self.load_balancer.record_request(provider, success)
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate a cache key for the request."""
        key_data = {
            "message": request.message,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "system_prompt": request.system_prompt,
            "conversation_history": request.conversation_history
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about providers."""
        stats = {
            "providers": [],
            "total_requests": sum(self.load_balancer.request_counts.values()),
            "total_failures": sum(self.load_balancer.failure_counts.values())
        }
        
        for provider_key, provider in self.providers.items():
            provider_stats = {
                "type": provider.provider_type,
                "healthy": provider.is_healthy,
                "enabled": provider.config.enabled,
                "priority": provider.config.priority,
                "requests": self.load_balancer.request_counts.get(provider_key, 0),
                "failures": self.load_balancer.failure_counts.get(provider_key, 0),
                "models": provider.config.models
            }
            stats["providers"].append(provider_stats)
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        
        for provider_key, provider in self.providers.items():
            try:
                health_status[provider_key] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {provider_key}: {e}")
                health_status[provider_key] = False
        
        return health_status
    
    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self.providers.values():
            try:
                await provider.close()
            except Exception as e:
                logger.error(f"Error closing provider: {e}")
        
        logger.info("LLM service closed")


# Provider factory functions
def create_openai_provider(config: Dict[str, Any]) -> OpenAIProvider:
    """Create OpenAI provider from configuration."""
    provider_config = ProviderConfig(
        provider_type=LLMProviderType.OPENAI,
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        models=config.get("models", ["gpt-4", "gpt-3.5-turbo"]),
        cost_per_1k_input_tokens=config.get("cost_per_1k_input_tokens", 0.03),
        cost_per_1k_output_tokens=config.get("cost_per_1k_output_tokens", 0.06),
        priority=config.get("priority", 1),
        enabled=config.get("enabled", True)
    )
    return OpenAIProvider(provider_config)


def create_anthropic_provider(config: Dict[str, Any]) -> AnthropicProvider:
    """Create Anthropic provider from configuration."""
    provider_config = ProviderConfig(
        provider_type=LLMProviderType.ANTHROPIC,
        api_key=config.get("api_key"),
        models=config.get("models", ["claude-2", "claude-instant-1"]),
        cost_per_1k_input_tokens=config.get("cost_per_1k_input_tokens", 0.008),
        cost_per_1k_output_tokens=config.get("cost_per_1k_output_tokens", 0.024),
        priority=config.get("priority", 2),
        enabled=config.get("enabled", True)
    )
    return AnthropicProvider(provider_config)


def create_ollama_provider(config: Dict[str, Any]) -> OllamaProvider:
    """Create Ollama provider from configuration."""
    provider_config = ProviderConfig(
        provider_type=LLMProviderType.OLLAMA,
        base_url=config.get("base_url", "http://localhost:11434"),
        models=config.get("models", ["llama3", "codellama", "mistral"]),
        cost_per_1k_input_tokens=0.0,  # Local model, no cost
        cost_per_1k_output_tokens=0.0,
        priority=config.get("priority", 3),
        enabled=config.get("enabled", True)
    )
    return OllamaProvider(provider_config)