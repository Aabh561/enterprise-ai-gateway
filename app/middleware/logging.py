"""
Logging Middleware for Enterprise AI Gateway

Provides request ID generation, structured logging, and request/response tracking.
"""

import time
import uuid
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging and request ID generation.
    
    This middleware:
    1. Generates unique request IDs for tracing
    2. Logs all incoming requests and outgoing responses
    3. Tracks request duration and status codes
    4. Provides structured logging for better observability
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and response with logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in the chain
            
        Returns:
            Response: HTTP response with added headers
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")
        
        # Log incoming request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=client_ip,
            user_agent=user_agent,
            headers=dict(request.headers) if logger.isEnabledFor("DEBUG") else None
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                client_ip=client_ip
            )
            
        except Exception as exc:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                duration_ms=round(duration * 1000, 2),
                client_ip=client_ip,
                error=str(exc),
                exc_info=True
            )
            
            raise
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request headers.
        
        Checks various headers in order of preference:
        1. X-Forwarded-For (from load balancers)
        2. X-Real-IP (from reverse proxies)
        3. Client host from request
        
        Args:
            request: HTTP request object
            
        Returns:
            str: Client IP address
        """
        # Check for forwarded IP headers (load balancers, proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP if multiple are present
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to client host
        return getattr(request.client, "host", "unknown") if request.client else "unknown"