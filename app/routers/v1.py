"""API v1 Router for Enterprise AI Gateway

This module contains all v1 API endpoints with advanced features.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user, get_request_id
from app.models import (
    ChatRequest,
    ChatResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StreamChunk,
)

# Note: to avoid importing heavy modules at import time, we import services lazily within functions.

router = APIRouter()

# Initialize services (would be done in app startup in production)
_llm_service = None
_cache_service = None
_vector_service = None


async def get_llm_service():
    """Get or initialize LLM service."""
    global _llm_service
    if _llm_service is None:
        from services.llm_service import (
            LLMService,
            create_ollama_provider,
            create_openai_provider,
        )

        _llm_service = LLMService()

        # Add providers
        openai_provider = create_openai_provider(
            {
                "api_key": "your-openai-key",  # Would come from config
                "enabled": True,
                "priority": 1,
            }
        )
        _llm_service.add_provider(openai_provider)

        ollama_provider = create_ollama_provider({"enabled": True, "priority": 2})
        _llm_service.add_provider(ollama_provider)

        await _llm_service.initialize()

    return _llm_service


async def get_cache_service():
    """Get or initialize caching service."""
    global _cache_service
    if _cache_service is None:
        from services.caching_service import CacheConfig, CachingService

        config = CacheConfig(
            ttl_seconds=3600, max_size_mb=100, redis_url="redis://localhost:6379/0"
        )
        _cache_service = CachingService(config)
        await _cache_service.initialize()

    return _cache_service


async def get_vector_service():
    """Get or initialize vector service."""
    global _vector_service
    if _vector_service is None:
        from services.vector_service import (
            ChunkingConfig,
            EmbeddingConfig,
            VectorDBConfig,
            VectorService,
        )

        vector_config = VectorDBConfig(
            url="http://localhost:8080", collection_name="Documents"
        )
        embedding_config = EmbeddingConfig()
        chunking_config = ChunkingConfig()

        _vector_service = VectorService(
            vector_config, embedding_config, chunking_config
        )
        await _vector_service.initialize()

    return _vector_service


@router.get("/")
async def root() -> Dict[str, Any]:
    """
    API root endpoint.

    Returns:
        Dict[str, Any]: API information
    """
    return {
        "message": "Enterprise AI Gateway API v1",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Advanced LLM Integration",
            "Multi-Provider Support",
            "Intelligent Caching",
            "RAG Pipeline",
            "Vector Search",
            "Document Processing",
        ],
    }


@router.get("/status")
async def status(
    current_user: Optional[dict] = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
) -> Dict[str, Any]:
    """
    Get API status with authentication and service health checks.

    Args:
        current_user: Current authenticated user
        request_id: Request ID for tracing

    Returns:
        Dict[str, Any]: Detailed status information
    """
    # Check service health
    service_health = {"llm": "unknown", "cache": "unknown", "vector_db": "unknown"}

    try:
        llm_service = await get_llm_service()
        llm_health = await llm_service.health_check()
        service_health["llm"] = "healthy" if any(llm_health.values()) else "unhealthy"
    except Exception:
        service_health["llm"] = "error"

    try:
        cache_service = await get_cache_service()
        cache_health = await cache_service.health_check()
        service_health["cache"] = (
            "healthy" if cache_health.get("overall") else "degraded"
        )
    except Exception:
        service_health["cache"] = "error"

    try:
        vector_service = await get_vector_service()
        vector_health = await vector_service.health_check()
        service_health["vector_db"] = (
            "healthy" if vector_health.get("overall") else "degraded"
        )
    except Exception:
        service_health["vector_db"] = "error"

    return {
        "status": "operational",
        "user": current_user,
        "request_id": request_id,
        "services": service_health,
        "timestamp": time.time(),
    }


@router.post("/chat/generate", response_model=ChatResponse)
async def chat_generate(
    request: ChatRequest,
    current_user: Optional[dict] = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
) -> ChatResponse:
    """
    Generate chat response using advanced LLM service with RAG support.

    Args:
        request: Chat generation request
        current_user: Current authenticated user
        request_id: Request ID for tracing

    Returns:
        ChatResponse: Generated response with metadata
    """
    start_time = time.time()

    try:
        # Get services
        llm_service = await get_llm_service()
        cache_service = await get_cache_service()
        vector_service = await get_vector_service() if request.use_rag else None

        # Prepare context from RAG if enabled
        rag_context = ""
        sources = []

        if request.use_rag and vector_service:
            try:
                # Search for relevant documents
                search_results = await vector_service.search(query=request.message, k=5)

                if search_results:
                    from services.vector_service import create_rag_context

                    rag_context = await create_rag_context(search_results)
                    sources = [
                        {
                            "title": r.chunk.metadata.title,
                            "score": r.similarity_score,
                            "content_preview": r.chunk.content[:200] + "...",
                        }
                        for r in search_results[:3]
                    ]
            except Exception as e:
                # RAG failure shouldn't block the request
                print(f"RAG search failed: {e}")

        # Create system prompt with RAG context
        system_prompt = "You are a helpful AI assistant."
        if rag_context:
            from services.vector_service import create_rag_prompt

            system_prompt = create_rag_prompt(
                request.message, rag_context, system_prompt
            )
            message = request.message  # Query is already in the prompt
        else:
            message = request.message

        # Create LLM request
        llm_request = LLMRequest(
            message=message if not rag_context else system_prompt,
            model=request.model or "gpt-3.5-turbo",
            provider=request.provider.value if request.provider else None,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            user_id=current_user.get("user_id") if current_user else "anonymous",
            request_id=request_id,
            system_prompt=system_prompt if not rag_context else None,
        )

        # Generate response
        llm_response = await llm_service.generate(llm_request)

        processing_time_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            response=llm_response.content,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            request_id=request_id,
            model=llm_response.model,
            provider=llm_response.provider.value,
            tokens_used=llm_response.usage.get("total_tokens", 0),
            processing_time_ms=processing_time_ms,
            cached=llm_response.cached,
            sources=sources if sources else None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: Optional[dict] = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
):
    """
    Stream chat response using advanced LLM service.

    Args:
        request: Chat streaming request
        current_user: Current authenticated user
        request_id: Request ID for tracing

    Returns:
        StreamingResponse: Streamed chat response
    """

    async def stream_generator():
        try:
            llm_service = await get_llm_service()

            # Create streaming LLM request
            llm_request = LLMRequest(
                message=request.message,
                model=request.model or "gpt-3.5-turbo",
                provider=request.provider.value if request.provider else None,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                user_id=current_user.get("user_id") if current_user else "anonymous",
                request_id=request_id,
            )

            # Stream response
            async for chunk in llm_service.stream_generate(llm_request):
                chunk_data = StreamChunk(chunk=chunk, done=False).json()
                yield f"data: {chunk_data}\n\n"

            # Send completion marker
            final_chunk = StreamChunk(
                chunk="", done=True, metadata={"request_id": request_id}
            ).json()
            yield f"data: {final_chunk}\n\n"

        except Exception as e:
            error_chunk = StreamChunk(chunk=f"Error: {str(e)}", done=True).json()
            yield f"data: {error_chunk}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = "default",
    current_user: Optional[dict] = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
) -> DocumentIngestResponse:
    """
    Upload and process documents for RAG using advanced vector service.

    Args:
        files: List of files to upload
        collection_name: Collection to store documents
        current_user: Current authenticated user
        request_id: Request ID for tracing

    Returns:
        DocumentIngestResponse: Upload results
    """
    start_time = time.time()

    try:
        vector_service = await get_vector_service()
        total_chunks = 0
        processed_files = []

        for file in files:
            # Validate file type
            allowed_types = ["pdf", "docx", "txt", "html", "md", "json"]
            file_ext = (
                file.filename.split(".")[-1].lower() if "." in file.filename else "txt"
            )

            if file_ext not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type '{file_ext}' not supported. Allowed: {allowed_types}",
                )

            # Read file content
            content = await file.read()

            # Determine document type
            doc_type_map = {
                "pdf": DocumentType.PDF,
                "docx": DocumentType.DOCX,
                "txt": DocumentType.TXT,
                "html": DocumentType.HTML,
                "md": DocumentType.MD,
                "json": DocumentType.JSON,
            }
            doc_type = doc_type_map.get(file_ext, DocumentType.TXT)

            # Create metadata
            metadata = DocumentMetadata(
                title=file.filename,
                file_size=len(content),
                document_type=doc_type,
                created_at=time.ctime(),
                custom_fields={"uploaded_by": current_user.get("user_id", "anonymous")},
            )

            # Process document
            result = await vector_service.ingest_document(
                content=content, document_type=doc_type, metadata=metadata
            )

            if result["success"]:
                total_chunks += result["chunks_created"]
                processed_files.append(
                    {
                        "filename": file.filename,
                        "size": len(content),
                        "type": file_ext,
                        "chunks": result["chunks_created"],
                    }
                )
            else:
                raise HTTPException(
                    status_code=500, detail=f"Failed to process file: {file.filename}"
                )

        processing_time_ms = (time.time() - start_time) * 1000

        return DocumentIngestResponse(
            ingested_count=len(processed_files),
            chunk_count=total_chunks,
            collection_name=collection_name,
            processing_time_ms=processing_time_ms,
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    current_user: Optional[dict] = Depends(get_current_user),
    request_id: str = Depends(get_request_id),
) -> SearchResponse:
    """
    Search documents using vector similarity.

    Args:
        request: Search request
        current_user: Current authenticated user
        request_id: Request ID for tracing

    Returns:
        SearchResponse: Search results
    """
    start_time = time.time()

    try:
        vector_service = await get_vector_service()

        # Perform vector search
        search_results = await vector_service.search(
            query=request.query,
            k=(request.limit or 5) + (request.offset or 0),
            collection_name=request.collection_name,
            filters=request.filters or {},
        )

        # Filter by threshold and apply pagination (offset, limit)
        filtered_results = [
            result
            for result in search_results
            if result.similarity_score >= (request.threshold or 0.0)
        ]

        start = request.offset or 0
        end = start + (request.limit or 5)
        page_results = filtered_results[start:end]

        # Convert to API response format
        results = []
        for result in page_results:
            search_result = SearchResult(
                content=result.chunk.content,
                score=result.similarity_score,
                metadata={
                    "title": result.chunk.metadata.title,
                    "chunk_index": result.chunk.chunk_index,
                    "document_type": result.chunk.metadata.document_type.value
                    if result.chunk.metadata.document_type
                    else None,
                    "file_path": result.chunk.metadata.file_path,
                    "rank": result.rank,
                },
            )
            results.append(search_result)

        processing_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(filtered_results),
            processing_time_ms=processing_time_ms,
            request_id=request_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Export router
__all__ = ["router"]
