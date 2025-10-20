"""
Enterprise AI Gateway - Main Application Entry Point

This module initializes the FastAPI application with all middleware,
exception handlers, and dependency injection configuration.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.exceptions import EnterpriseAIException, ValidationError, AuthenticationError
from app.middleware.logging import LoggingMiddleware
from app.middleware.security import SecurityMiddleware
from app.routers import v1
from app.dependencies import get_current_user

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration in seconds'
)
LLM_REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['provider', 'model', 'status']
)
CACHE_HIT_COUNT = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.
    """
    # Startup
    logger.info("Starting Enterprise AI Gateway", version=settings.app.version)
    
    # Initialize core application state
    app.state.startup_time = time.time()
    logger.info("Enterprise AI Gateway started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enterprise AI Gateway")
    
    # Cleanup services
    if hasattr(app.state, 'cache_service'):
        await app.state.cache_service.close()
    
    if hasattr(app.state, 'chat_service'):
        await app.state.chat_service.close()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    
    app = FastAPI(
        title="Enterprise AI Gateway",
        description="The Central Nervous System for Enterprise AI - A secure, intelligent, and observable platform for LLM interactions",
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.app.debug else None,
        redoc_url="/redoc" if settings.app.debug else None,
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(v1.router, prefix="/api/v1")
    
    # Add health and metrics endpoints
    setup_health_endpoints(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    
    # Security middleware (first for security checks)
    app.add_middleware(SecurityMiddleware)
    
    # Trusted host middleware
    if not settings.app.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure based on deployment
        )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors.origins,
        allow_credentials=settings.api.cors.allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Request ID and logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Prometheus metrics middleware
    app.add_middleware(MetricsMiddleware)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting Prometheus metrics.
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        return response


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Configure global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(EnterpriseAIException)
    async def enterprise_ai_exception_handler(request: Request, exc: EnterpriseAIException):
        logger.error(
            "Enterprise AI Exception",
            request_id=getattr(request.state, 'request_id', None),
            error_code=exc.error_code,
            error_message=str(exc),
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": str(exc),
                    "request_id": getattr(request.state, 'request_id', None)
                }
            }
        )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        logger.warning(
            "Validation Error",
            request_id=getattr(request.state, 'request_id', None),
            errors=exc.errors,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed",
                    "details": exc.errors,
                    "request_id": getattr(request.state, 'request_id', None)
                }
            }
        )
    
    @app.exception_handler(AuthenticationError)
    async def auth_exception_handler(request: Request, exc: AuthenticationError):
        logger.warning(
            "Authentication Error",
            request_id=getattr(request.state, 'request_id', None),
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "AUTHENTICATION_ERROR",
                    "message": "Authentication failed",
                    "request_id": getattr(request.state, 'request_id', None)
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        logger.error(
            "Unhandled Exception",
            request_id=request_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            exc_info=True
        )
        
        if settings.app.debug:
            # Return detailed error in debug mode
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                        "request_id": request_id
                    }
                }
            )
        else:
            # Return generic error in production
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred",
                        "request_id": request_id
                    }
                }
            )


def setup_health_endpoints(app: FastAPI) -> None:
    """
    Configure health check and metrics endpoints.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.get("/health")
    async def health_check():
        """
        Health check endpoint for load balancers and monitoring.
        """
        uptime = time.time() - getattr(app.state, 'startup_time', time.time())
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "version": settings.app.version,
            "environment": settings.app.environment
        }
    
    @app.get("/health/ready")
    async def readiness_check():
        """
        Readiness check endpoint for Kubernetes.
        """
        try:
            # Check critical services
            if hasattr(app.state, 'cache_service'):
                await app.state.cache_service.health_check()
            
            return {"status": "ready"}
        except Exception as e:
            logger.error("Readiness check failed", error=str(e))
            raise HTTPException(status_code=503, detail="Service not ready")
    
    @app.get("/metrics")
    async def metrics():
        """
        Prometheus metrics endpoint.
        """
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


# Create the application instance
app = create_app()

# Export for ASGI servers
__all__ = ["app"]