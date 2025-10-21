"""
Security Middleware for Enterprise AI Gateway

Provides API key validation, rate limiting, and basic security checks.
"""

import time
from typing import Callable, Optional

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings
from app.exceptions import AuthenticationError, RateLimitExceededError, SecurityError

logger = structlog.get_logger()
settings = get_settings()


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for Enterprise AI Gateway.

    This middleware provides:
    1. API key validation
    2. Basic rate limiting
    3. Security header validation
    4. Request size limits
    """

    # Paths that don't require authentication
    EXEMPT_PATHS = {
        "/health",
        "/health/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(self, app):
        super().__init__(app)
        self.rate_limit_store = {}  # Simple in-memory store for demo

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process security checks for the request.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in the chain

        Returns:
            Response: HTTP response or security error response
        """
        try:
            # Skip security checks for exempt paths
            if request.url.path in self.EXEMPT_PATHS:
                return await call_next(request)

            # Check request size
            await self._check_request_size(request)

            # Check rate limits
            await self._check_rate_limit(request)

            # Validate API key
            await self._validate_api_key(request)

            # Check security headers
            await self._check_security_headers(request)

            # Process request
            response = await call_next(request)

            # Add security headers to response
            self._add_security_headers(response)

            # Add rate limit headers if available
            if hasattr(request.state, "rate_limit_limit"):
                response.headers["X-RateLimit-Limit"] = str(
                    request.state.rate_limit_limit
                )
                response.headers["X-RateLimit-Remaining"] = str(
                    request.state.rate_limit_remaining
                )

            return response

        except (AuthenticationError, RateLimitExceededError, SecurityError) as e:
            # Return structured error response
            return JSONResponse(status_code=e.status_code, content=e.to_dict())
        except Exception as e:
            logger.error(
                "Security middleware error",
                error=str(e),
                request_id=getattr(request.state, "request_id", None),
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "SECURITY_ERROR",
                        "message": "Security check failed",
                    }
                },
            )

    async def _check_request_size(self, request: Request) -> None:
        """
        Check if request content length is within limits.

        Args:
            request: HTTP request to check

        Raises:
            SecurityError: If request is too large
        """
        content_length = request.headers.get("content-length")
        if content_length:
            size = int(content_length)
            max_size = 10 * 1024 * 1024  # 10MB limit

            if size > max_size:
                logger.warning(
                    "Request too large",
                    content_length=size,
                    max_size=max_size,
                    client_ip=self._get_client_ip(request),
                )
                raise SecurityError(
                    "Request too large",
                    security_check="content_length",
                    client_ip=self._get_client_ip(request),
                )

    async def _check_rate_limit(self, request: Request) -> None:
        """
        Check rate limits for the client.

        Args:
            request: HTTP request to check

        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        if not settings.rate_limiting.enabled:
            return

        client_ip = self._get_client_ip(request)
        api_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization")
            or "anonymous"
        )
        if api_key.startswith("Bearer "):
            api_key = api_key[7:]
        current_time = time.time()
        window = 60  # 1 minute window

        # Clean old entries
        cutoff_time = current_time - window
        self.rate_limit_store = {
            key: timestamps
            for key, timestamps in self.rate_limit_store.items()
            if any(ts > cutoff_time for ts in timestamps)
        }

        # Compose key combining API key and IP for fairness across keys
        bucket_key = f"{api_key}:{client_ip}"

        # Get or create entry for this bucket
        if bucket_key not in self.rate_limit_store:
            self.rate_limit_store[bucket_key] = []

        # Filter to current window
        self.rate_limit_store[bucket_key] = [
            ts for ts in self.rate_limit_store[bucket_key] if ts > cutoff_time
        ]

        # Check rate limit
        current_requests = len(self.rate_limit_store[bucket_key])
        if current_requests >= settings.rate_limiting.per_minute:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                current_requests=current_requests,
                limit=settings.rate_limiting.per_minute,
            )
            raise RateLimitExceededError(
                f"Rate limit exceeded: {current_requests}/{settings.rate_limiting.per_minute} requests per minute",
                retry_after=60,
            )

        # Record this request
        self.rate_limit_store[bucket_key].append(current_time)

        # Expose rate limit info on request for response headers
        request.state.rate_limit_limit = settings.rate_limiting.per_minute
        request.state.rate_limit_remaining = max(
            0,
            settings.rate_limiting.per_minute - len(self.rate_limit_store[bucket_key]),
        )

    async def _validate_api_key(self, request: Request) -> None:
        """
        Validate API key from request headers.

        Args:
            request: HTTP request to validate

        Raises:
            AuthenticationError: If API key is invalid or missing
        """
        if not settings.security.api_keys.enabled:
            return

        # Get API key from header
        api_key = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization"
        )

        if not api_key:
            raise AuthenticationError("API key required")

        # Remove "Bearer " prefix if present
        if api_key.startswith("Bearer "):
            api_key = api_key[7:]

        # Validation supporting rotation list
        valid_keys = set(settings.security.api_keys.secrets or [])
        # Include single secret for backward compatibility
        if settings.security.api_keys.secret:
            valid_keys.add(settings.security.api_keys.secret)

        if api_key not in valid_keys:
            logger.warning(
                "Invalid API key",
                client_ip=self._get_client_ip(request),
                provided_key_prefix=api_key[:8] + "..."
                if len(api_key) > 8
                else api_key,
            )
            raise AuthenticationError("Invalid API key")

        # Store validated API key info in request state
        request.state.api_key_valid = True
        request.state.api_key = api_key

    async def _check_security_headers(self, request: Request) -> None:
        """
        Check for required security headers.

        Args:
            request: HTTP request to check

        Raises:
            SecurityError: If required security headers are missing
        """
        # Check for HTTPS in production
        if (
            not settings.app.debug
            and settings.security.api_keys.require_https
            and request.url.scheme != "https"
        ):
            raise SecurityError("HTTPS required", security_check="https_required")

        # Check for suspicious user agents (basic example)
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_patterns = ["bot", "crawler", "spider", "scraper"]

        if any(pattern in user_agent for pattern in suspicious_patterns):
            logger.info(
                "Suspicious user agent detected",
                user_agent=user_agent,
                client_ip=self._get_client_ip(request),
            )

    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to the response.

        Args:
            response: HTTP response to modify
        """
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        if not settings.app.debug:
            response.headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains"

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.

        Args:
            request: HTTP request object

        Returns:
            str: Client IP address
        """
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return (
            getattr(request.client, "host", "unknown") if request.client else "unknown"
        )
