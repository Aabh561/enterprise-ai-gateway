"""
Pydantic Models for Enterprise AI Gateway API

This module defines all request/response schemas using Pydantic models.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime


class LLMProvider(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Individual chat message."""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat generation request schema."""
    message: str = Field(..., min_length=1, max_length=8192, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    provider: Optional[LLMProvider] = Field(None, description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens: Optional[int] = Field(1024, ge=1, le=4096, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Whether to stream the response")
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    plugins: Optional[List[str]] = Field(None, description="Plugins to enable for this request")
    
    @validator('message')
    def validate_message(cls, v):
        """Validate message content."""
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()


class ChatResponse(BaseModel):
    """Chat generation response schema."""
    response: str = Field(..., description="AI-generated response")
    conversation_id: str = Field(..., description="Conversation ID")
    request_id: str = Field(..., description="Request ID for tracing")
    model: str = Field(..., description="Model used for generation")
    provider: str = Field(..., description="Provider used")
    tokens_used: int = Field(..., description="Total tokens consumed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cached: bool = Field(False, description="Whether response was cached")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="RAG sources used")
    plugin_results: Optional[Dict[str, Any]] = Field(None, description="Plugin execution results")


class StreamChunk(BaseModel):
    """Individual streaming response chunk."""
    chunk: str = Field(..., description="Text chunk")
    done: bool = Field(False, description="Whether streaming is complete")
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: Dict[str, Any] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed",
                    "request_id": "12345678-1234-5678-9012-123456789012"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Response timestamp")
    uptime_seconds: float = Field(..., description="Service uptime")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")


class StatusResponse(BaseModel):
    """Detailed status response schema."""
    status: str = Field(..., description="Overall status")
    user: Optional[Dict[str, Any]] = Field(None, description="Current user info")
    request_id: str = Field(..., description="Request ID")
    services: Dict[str, str] = Field(..., description="Service statuses")
    system_info: Optional[Dict[str, Any]] = None


class DocumentIngestRequest(BaseModel):
    """Document ingestion request schema."""
    documents: List[Dict[str, Any]] = Field(..., description="Documents to ingest")
    collection_name: Optional[str] = Field("default", description="Collection name")
    chunk_size: Optional[int] = Field(512, ge=100, le=2048, description="Text chunk size")
    chunk_overlap: Optional[int] = Field(50, ge=0, le=500, description="Chunk overlap size")
    metadata: Optional[Dict[str, Any]] = None


class DocumentIngestResponse(BaseModel):
    """Document ingestion response schema."""
    ingested_count: int = Field(..., description="Number of documents ingested")
    chunk_count: int = Field(..., description="Number of chunks created")
    collection_name: str = Field(..., description="Collection name")
    processing_time_ms: float = Field(..., description="Processing time")
    request_id: str = Field(..., description="Request ID")


class SearchRequest(BaseModel):
    """Vector search request schema."""
    query: str = Field(..., min_length=1, max_length=1024, description="Search query")
    collection_name: Optional[str] = Field("default", description="Collection to search")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Maximum results")
    threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_metadata: bool = Field(True, description="Include document metadata")


class SearchResult(BaseModel):
    """Individual search result."""
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")


class SearchResponse(BaseModel):
    """Vector search response schema."""
    results: List[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total results found")
    processing_time_ms: float = Field(..., description="Processing time")
    request_id: str = Field(..., description="Request ID")


class PluginExecutionRequest(BaseModel):
    """Plugin execution request schema."""
    plugin_name: str = Field(..., description="Plugin to execute")
    action: str = Field(..., description="Action to perform")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Execution timeout")


class PluginExecutionResponse(BaseModel):
    """Plugin execution response schema."""
    result: Any = Field(..., description="Plugin execution result")
    plugin_name: str = Field(..., description="Executed plugin")
    action: str = Field(..., description="Executed action")
    success: bool = Field(..., description="Execution success")
    execution_time_ms: float = Field(..., description="Execution time")
    request_id: str = Field(..., description="Request ID")


# Export all models
__all__ = [
    "LLMProvider",
    "MessageRole", 
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "StreamChunk",
    "ErrorResponse",
    "HealthResponse",
    "StatusResponse",
    "DocumentIngestRequest",
    "DocumentIngestResponse",
    "SearchRequest",
    "SearchResult", 
    "SearchResponse",
    "PluginExecutionRequest",
    "PluginExecutionResponse"
]