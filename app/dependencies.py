"""
Dependency Injection for Enterprise AI Gateway

This module provides dependency injection functions for FastAPI endpoints.
"""

from typing import Optional, Any
from fastapi import Depends, HTTPException, Request, status

from app.config import get_settings
from app.exceptions import AuthenticationError

settings = get_settings()


async def get_current_user(request: Request) -> Optional[dict]:
    """
    Get current user information from request.
    
    This is a placeholder implementation. In production, this would
    validate JWT tokens and return user information.
    
    Args:
        request: HTTP request object
        
    Returns:
        Optional[dict]: User information if authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check if API key is valid (set by security middleware)
    if not getattr(request.state, 'api_key_valid', False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Return basic user info (in production, decode from JWT)
    return {
        "user_id": "system",
        "username": "api_user",
        "permissions": ["chat", "admin"]
    }


async def get_request_id(request: Request) -> str:
    """
    Get request ID from request state.
    
    Args:
        request: HTTP request object
        
    Returns:
        str: Request ID
    """
    return getattr(request.state, 'request_id', 'unknown')


async def get_settings_dependency() -> Any:
    """
    Get application settings as a dependency.
    
    Returns:
        Settings: Application settings
    """
    return get_settings()


# Export dependencies
__all__ = ["get_current_user", "get_request_id", "get_settings_dependency"]