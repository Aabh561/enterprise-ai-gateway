"""
Custom Exception Classes for Enterprise AI Gateway

This module defines all custom exceptions used throughout the application
with proper error codes and HTTP status mappings.
"""

from typing import Any, Dict, List, Optional


class EnterpriseAIException(Exception):
    """
    Base exception class for all Enterprise AI Gateway exceptions.

    All custom exceptions should inherit from this class to ensure
    consistent error handling and logging.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


class ValidationError(EnterpriseAIException):
    """Exception raised when input validation fails."""

    def __init__(
        self,
        message: str = "Input validation failed",
        errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            message=message, error_code="VALIDATION_ERROR", status_code=422
        )
        self.errors = errors or []


class AuthenticationError(EnterpriseAIException):
    """Exception raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message, error_code="AUTHENTICATION_ERROR", status_code=401
        )


class AuthorizationError(EnterpriseAIException):
    """Exception raised when authorization fails."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message, error_code="AUTHORIZATION_ERROR", status_code=403
        )


class RateLimitExceededError(EnterpriseAIException):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None
    ):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details,
        )


class LLMProviderError(EnterpriseAIException):
    """Exception raised when LLM provider encounters an error."""

    def __init__(
        self,
        message: str,
        provider: str,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "provider": provider,
            "model": model,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(
            message=message,
            error_code="LLM_PROVIDER_ERROR",
            status_code=502,
            details=details,
        )


class VectorDatabaseError(EnterpriseAIException):
    """Exception raised when vector database operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "operation": operation,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(
            message=message,
            error_code="VECTOR_DATABASE_ERROR",
            status_code=502,
            details=details,
        )


class CacheError(EnterpriseAIException):
    """Exception raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "cache_key": cache_key,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(
            message=message, error_code="CACHE_ERROR", status_code=500, details=details
        )


class PluginError(EnterpriseAIException):
    """Exception raised when plugin operations fail."""

    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        plugin_action: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "plugin_name": plugin_name,
            "plugin_action": plugin_action,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(
            message=message, error_code="PLUGIN_ERROR", status_code=500, details=details
        )


class PluginExecutionError(PluginError):
    """Exception raised when plugin execution fails."""

    def __init__(
        self,
        message: str,
        plugin_name: str,
        execution_time: Optional[float] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "plugin_name": plugin_name,
            "execution_time": execution_time,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(
            message=message,
            error_code="PLUGIN_EXECUTION_ERROR",
            status_code=500,
            details=details,
        )


class PluginTimeoutError(PluginError):
    """Exception raised when plugin execution times out."""

    def __init__(
        self,
        message: str = "Plugin execution timed out",
        plugin_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        details = {"plugin_name": plugin_name, "timeout_seconds": timeout_seconds}
        super().__init__(
            message=message,
            error_code="PLUGIN_TIMEOUT_ERROR",
            status_code=408,
            details=details,
        )


class SecurityError(EnterpriseAIException):
    """Exception raised when security violations are detected."""

    def __init__(
        self,
        message: str,
        security_check: Optional[str] = None,
        client_ip: Optional[str] = None,
    ):
        details = {"security_check": security_check, "client_ip": client_ip}
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            status_code=403,
            details=details,
        )


class PIIDetectionError(SecurityError):
    """Exception raised when PII detection/redaction fails."""

    def __init__(
        self,
        message: str = "PII detection failed",
        original_error: Optional[Exception] = None,
    ):
        details = {"original_error": str(original_error) if original_error else None}
        super().__init__(
            message=message,
            error_code="PII_DETECTION_ERROR",
            status_code=500,
            details=details,
        )


class FileProcessingError(EnterpriseAIException):
    """Exception raised when file processing fails."""

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "filename": filename,
            "file_type": file_type,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR",
            status_code=422,
            details=details,
        )


class ConfigurationError(EnterpriseAIException):
    """Exception raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
    ):
        details = {"config_section": config_section, "config_key": config_key}
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )


class ServiceUnavailableError(EnterpriseAIException):
    """Exception raised when a required service is unavailable."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        details = {"service_name": service_name, "retry_after": retry_after}
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=details,
        )


class ContentTooLargeError(EnterpriseAIException):
    """Exception raised when content exceeds size limits."""

    def __init__(
        self,
        message: str = "Content too large",
        max_size: Optional[int] = None,
        actual_size: Optional[int] = None,
    ):
        details = {"max_size": max_size, "actual_size": actual_size}
        super().__init__(
            message=message,
            error_code="CONTENT_TOO_LARGE",
            status_code=413,
            details=details,
        )


class NotFoundError(EnterpriseAIException):
    """Exception raised when a resource is not found."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        details = {"resource_type": resource_type, "resource_id": resource_id}
        super().__init__(
            message=message, error_code="NOT_FOUND", status_code=404, details=details
        )


class DatabaseError(EnterpriseAIException):
    """Exception raised when database operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "operation": operation,
            "table": table,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details,
        )


# Export all exceptions
__all__ = [
    "EnterpriseAIException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitExceededError",
    "LLMProviderError",
    "VectorDatabaseError",
    "CacheError",
    "PluginError",
    "PluginExecutionError",
    "PluginTimeoutError",
    "SecurityError",
    "PIIDetectionError",
    "FileProcessingError",
    "ConfigurationError",
    "ServiceUnavailableError",
    "ContentTooLargeError",
    "NotFoundError",
    "DatabaseError",
]
