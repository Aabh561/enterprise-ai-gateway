"""
Enterprise AI Gateway - Application Package

This package contains the main FastAPI application and its core components.
"""

__version__ = "0.1.0"
__author__ = "Enterprise AI Team"
__email__ = "team@enterprise-ai.com"

from app.main import app

__all__ = ["app"]