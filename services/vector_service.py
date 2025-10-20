"""
Advanced Vector Database Service for Enterprise AI Gateway

Provides sophisticated RAG pipeline with Weaviate/Milvus integration,
advanced chunking strategies, multi-modal document processing, and semantic search.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncIterator
from urllib.parse import urlparse

import aiofiles
import httpx
import numpy as np
import structlog
from prometheus_client import Counter, Histogram, Gauge

# Document processing imports
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from unstructured.documents.elements import Element
    from unstructured.partition.auto import partition
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

# Vector database imports
try:
    import weaviate
    HAS_WEAVIATE = True
except ImportError:
    HAS_WEAVIATE = False

try:
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False

logger = structlog.get_logger(__name__)

# Metrics
VECTOR_OPERATIONS_COUNTER = Counter(
    'vector_operations_total',
    'Total vector database operations',
    ['operation', 'collection', 'status']
)

VECTOR_SEARCH_LATENCY = Histogram(
    'vector_search_duration_seconds',
    'Vector search duration',
    ['collection', 'k']
)

DOCUMENT_PROCESSING_COUNTER = Counter(
    'document_processing_total',
    'Total documents processed',
    ['doc_type', 'status']
)

CHUNK_GENERATION_COUNTER = Counter(
    'chunks_generated_total',
    'Total chunks generated',
    ['strategy']
)

VECTOR_DB_SIZE_GAUGE = Gauge(
    'vector_db_documents_total',
    'Total documents in vector database',
    ['collection']
)


class VectorDBProvider(str, Enum):
    """Supported vector database providers."""
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    CHROMA = "chroma"


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MD = "markdown"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 2048
    sentence_split_threshold: float = 0.5
    paragraph_split_threshold: float = 0.7
    semantic_threshold: float = 0.8
    preserve_metadata: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    device: str = "cpu"  # "cpu" or "cuda"


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    provider: VectorDBProvider = VectorDBProvider.WEAVIATE
    url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    collection_name: str = "Documents"
    
    # Weaviate specific
    weaviate_auth_config: Optional[Dict[str, Any]] = None
    
    # Milvus specific
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    
    # Search settings
    similarity_metric: str = "cosine"
    index_type: str = "HNSW"
    search_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    """Document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    document_type: Optional[DocumentType] = None
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """A chunk of a processed document."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    tokens_count: int = 0
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata.__dict__,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "tokens_count": self.tokens_count,
            "relevance_score": self.relevance_score
        }


@dataclass
class SearchResult:
    """Vector search result."""
    chunk: DocumentChunk
    similarity_score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": self.similarity_score,
            "rank": self.rank
        }


class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._model_loaded = False
    
    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._model_loaded:
            return
        
        try:
            # Load model in executor to avoid blocking
            def load_model():
                import importlib
                ST = importlib.import_module("sentence_transformers").SentenceTransformer
                return ST(self.config.model_name, device=self.config.device)
            self.model = await asyncio.get_event_loop().run_in_executor(
                None,
                load_model
            )
            self._model_loaded = True
            logger.info(f"Loaded embedding model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode text(s) to embeddings."""
        if not self._model_loaded:
            await self.initialize()
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Encode in batches
            embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.encode(
                        batch,
                        normalize_embeddings=self.config.normalize_embeddings,
                        convert_to_tensor=False
                    ).tolist()
                )
                embeddings.extend(batch_embeddings)
            
            return embeddings[0] if len(texts) == 1 else embeddings
        
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            raise
    
    async def encode_query(self, query: str) -> List[float]:
        """Encode a search query."""
        embedding = await self.encode(query)
        return embedding if isinstance(embedding[0], float) else embedding[0]


class DocumentProcessor:
    """Advanced document processor with multi-modal support."""
    
    def __init__(self, chunking_config: ChunkingConfig):
        self.chunking_config = chunking_config
    
    async def process_document(
        self,
        content: Union[str, bytes],
        document_type: DocumentType,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[DocumentChunk]:
        """Process a document into chunks."""
        try:
            # Extract text based on document type
            if isinstance(content, bytes):
                text = await self._extract_text_from_bytes(content, document_type)
            else:
                text = content
            
            # Create chunks
            chunks = await self._create_chunks(text, metadata or DocumentMetadata())
            
            DOCUMENT_PROCESSING_COUNTER.labels(
                doc_type=document_type,
                status="success"
            ).inc()
            
            return chunks
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            DOCUMENT_PROCESSING_COUNTER.labels(
                doc_type=document_type,
                status="error"
            ).inc()
            raise
    
    async def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a file from disk."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine document type from extension
        doc_type = self._get_document_type(file_path.suffix.lower())
        
        # Create metadata
        metadata = DocumentMetadata(
            title=file_path.stem,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            document_type=doc_type,
            created_at=time.ctime(file_path.stat().st_ctime),
            modified_at=time.ctime(file_path.stat().st_mtime)
        )
        
        # Read file content
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
        
        return await self.process_document(content, doc_type, metadata)
    
    async def process_url(self, url: str) -> List[DocumentChunk]:
        """Process a document from URL."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Determine content type
                content_type = response.headers.get("content-type", "").lower()
                doc_type = self._get_document_type_from_content_type(content_type)
                
                # Create metadata
                metadata = DocumentMetadata(
                    title=self._extract_title_from_url(url),
                    source_url=url,
                    document_type=doc_type,
                    file_size=len(response.content)
                )
                
                return await self.process_document(response.content, doc_type, metadata)
                
        except Exception as e:
            logger.error(f"URL processing failed for {url}: {e}")
            raise
    
    async def _extract_text_from_bytes(self, content: bytes, doc_type: DocumentType) -> str:
        """Extract text from bytes based on document type."""
        if doc_type == DocumentType.TXT:
            return content.decode('utf-8', errors='ignore')
        
        elif doc_type == DocumentType.PDF:
            if HAS_PYMUPDF:
                return await self._extract_pdf_text_pymupdf(content)
            elif HAS_UNSTRUCTURED:
                return await self._extract_with_unstructured(content, doc_type)
            else:
                raise ImportError("PDF processing requires PyMuPDF or unstructured")
        
        elif doc_type in [DocumentType.HTML, DocumentType.MD, DocumentType.JSON]:
            if HAS_UNSTRUCTURED:
                return await self._extract_with_unstructured(content, doc_type)
            else:
                return content.decode('utf-8', errors='ignore')
        
        else:
            # Fallback to text extraction
            return content.decode('utf-8', errors='ignore')
    
    async def _extract_pdf_text_pymupdf(self, content: bytes) -> str:
        """Extract text from PDF using PyMuPDF."""
        def extract_sync():
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        
        return await asyncio.get_event_loop().run_in_executor(None, extract_sync)
    
    async def _extract_with_unstructured(self, content: bytes, doc_type: DocumentType) -> str:
        """Extract text using unstructured library."""
        def extract_sync():
            # Save content to temp file for unstructured
            import tempfile
            import os
            
            suffix = f".{doc_type.value}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                elements = partition(tmp_path)
                text = "\n".join([str(element) for element in elements])
                return text
            finally:
                os.unlink(tmp_path)
        
        return await asyncio.get_event_loop().run_in_executor(None, extract_sync)
    
    async def _create_chunks(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Create chunks from text using the configured strategy."""
        strategy = self.chunking_config.strategy
        
        if strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = await self._chunk_fixed_size(text)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = await self._chunk_by_sentences(text)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            chunks = await self._chunk_by_paragraphs(text)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = await self._chunk_semantic(text)
        elif strategy == ChunkingStrategy.HYBRID:
            chunks = await self._chunk_hybrid(text)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = await self._chunk_sliding_window(text)
        else:
            chunks = await self._chunk_fixed_size(text)
        
        # Create DocumentChunk objects
        doc_chunks = []
        for i, (chunk_text, start_char, end_char) in enumerate(chunks):
            chunk_id = self._generate_chunk_id(metadata, i)
            doc_chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_text,
                metadata=metadata,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                tokens_count=len(chunk_text.split())
            )
            doc_chunks.append(doc_chunk)
        
        CHUNK_GENERATION_COUNTER.labels(strategy=strategy).inc(len(doc_chunks))
        
        return doc_chunks
    
    async def _chunk_fixed_size(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text into fixed-size pieces with overlap."""
        chunks = []
        chunk_size = self.chunking_config.chunk_size
        overlap = self.chunking_config.chunk_overlap
        
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Look for last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.chunking_config.min_chunk_size:
                chunks.append((chunk_text, start, end))
            
            start = max(start + 1, end - overlap)
        
        return chunks
    
    async def _chunk_by_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by sentences."""
        # Simple sentence splitting (could be improved with spaCy or NLTK)
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.chunking_config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_end = current_start + len(current_chunk)
                chunks.append((current_chunk, current_start, chunk_end))
                current_chunk = sentence
                current_start = chunk_end
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk:
            chunk_end = current_start + len(current_chunk)
            chunks.append((current_chunk, current_start, chunk_end))
        
        return chunks
    
    async def _chunk_by_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by paragraphs."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) > self.chunking_config.chunk_size and current_chunk:
                chunk_end = current_start + len(current_chunk)
                chunks.append((current_chunk, current_start, chunk_end))
                current_chunk = paragraph
                current_start = chunk_end
            else:
                current_chunk = potential_chunk
        
        if current_chunk:
            chunk_end = current_start + len(current_chunk)
            chunks.append((current_chunk, current_start, chunk_end))
        
        return chunks
    
    async def _chunk_semantic(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text based on semantic similarity (placeholder implementation)."""
        # This is a simplified semantic chunking
        # In practice, you'd use embeddings to determine semantic boundaries
        return await self._chunk_by_sentences(text)
    
    async def _chunk_hybrid(self, text: str) -> List[Tuple[str, int, int]]:
        """Hybrid chunking combining multiple strategies."""
        # Start with paragraph chunks
        paragraph_chunks = await self._chunk_by_paragraphs(text)
        
        final_chunks = []
        for chunk_text, start, end in paragraph_chunks:
            # If chunk is too large, further split by sentences
            if len(chunk_text) > self.chunking_config.max_chunk_size:
                sentence_chunks = await self._chunk_by_sentences(chunk_text)
                for sent_text, sent_start, sent_end in sentence_chunks:
                    final_chunks.append((sent_text, start + sent_start, start + sent_end))
            else:
                final_chunks.append((chunk_text, start, end))
        
        return final_chunks
    
    async def _chunk_sliding_window(self, text: str) -> List[Tuple[str, int, int]]:
        """Sliding window chunking."""
        chunks = []
        window_size = self.chunking_config.chunk_size
        step_size = window_size - self.chunking_config.chunk_overlap
        
        for start in range(0, len(text), step_size):
            end = min(start + window_size, len(text))
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.chunking_config.min_chunk_size:
                chunks.append((chunk_text, start, end))
            
            if end >= len(text):
                break
        
        return chunks
    
    def _generate_chunk_id(self, metadata: DocumentMetadata, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        base_data = f"{metadata.title}_{metadata.file_path}_{chunk_index}"
        return hashlib.md5(base_data.encode()).hexdigest()[:16]
    
    def _get_document_type(self, extension: str) -> DocumentType:
        """Get document type from file extension."""
        ext_map = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.txt': DocumentType.TXT,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.md': DocumentType.MD,
            '.json': DocumentType.JSON,
            '.csv': DocumentType.CSV,
            '.xlsx': DocumentType.XLSX
        }
        return ext_map.get(extension, DocumentType.TXT)
    
    def _get_document_type_from_content_type(self, content_type: str) -> DocumentType:
        """Get document type from HTTP content type."""
        if 'pdf' in content_type:
            return DocumentType.PDF
        elif 'html' in content_type:
            return DocumentType.HTML
        elif 'json' in content_type:
            return DocumentType.JSON
        else:
            return DocumentType.TXT
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract title from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        if path:
            return path.split('/')[-1]
        return parsed.netloc


class BaseVectorDB(ABC):
    """Abstract base class for vector databases."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database connection."""
        pass
    
    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata_schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a collection/index."""
        pass
    
    @abstractmethod
    async def insert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Insert document chunks."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        collection_name: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_by_id(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by IDs."""
        pass
    
    @abstractmethod
    async def update_chunk(self, chunk: DocumentChunk) -> bool:
        """Update a chunk."""
        pass
    
    @abstractmethod
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        pass
    
    async def close(self) -> None:
        """Close the database connection."""
        pass


class WeaviateVectorDB(BaseVectorDB):
    """Weaviate vector database implementation."""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize Weaviate client."""
        if not HAS_WEAVIATE:
            raise ImportError("Weaviate client not available. Install with: pip install weaviate-client")
        
        try:
            auth_config = None
            if self.config.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.config.api_key)
            elif self.config.weaviate_auth_config:
                # Handle other auth types
                pass
            
            self.client = weaviate.Client(
                url=self.config.url,
                auth_client_secret=auth_config
            )
            
            # Test connection
            if not self.client.is_ready():
                raise ConnectionError("Weaviate is not ready")
            
            self._initialized = True
            logger.info("Weaviate client initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise
    
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata_schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a Weaviate class (collection)."""
        try:
            # Define class schema
            class_schema = {
                "class": collection_name,
                "description": f"Document chunks for {collection_name}",
                "vectorizer": "none",  # We provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The chunk content"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Index of chunk in document"
                    },
                    {
                        "name": "start_char",
                        "dataType": ["int"],
                        "description": "Start character position"
                    },
                    {
                        "name": "end_char",
                        "dataType": ["int"],
                        "description": "End character position"
                    },
                    {
                        "name": "tokens_count",
                        "dataType": ["int"],
                        "description": "Number of tokens"
                    },
                    {
                        "name": "title",
                        "dataType": ["string"],
                        "description": "Document title"
                    },
                    {
                        "name": "author",
                        "dataType": ["string"],
                        "description": "Document author"
                    },
                    {
                        "name": "source_url",
                        "dataType": ["string"],
                        "description": "Source URL"
                    },
                    {
                        "name": "file_path",
                        "dataType": ["string"],
                        "description": "File path"
                    },
                    {
                        "name": "document_type",
                        "dataType": ["string"],
                        "description": "Document type"
                    },
                    {
                        "name": "created_at",
                        "dataType": ["string"],
                        "description": "Creation timestamp"
                    }
                ]
            }
            
            # Add custom metadata fields
            if metadata_schema:
                for field_name, field_type in metadata_schema.items():
                    class_schema["properties"].append({
                        "name": field_name,
                        "dataType": [field_type],
                        "description": f"Custom field: {field_name}"
                    })
            
            # Create class
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.schema.create_class(class_schema)
            )
            
            logger.info(f"Created Weaviate class: {collection_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create Weaviate class: {e}")
            return False
    
    async def insert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Insert chunks into Weaviate."""
        try:
            def insert_sync():
                with self.client.batch as batch:
                    batch.batch_size = 100
                    
                    for chunk in chunks:
                        data_object = {
                            "content": chunk.content,
                            "chunk_index": chunk.chunk_index,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "tokens_count": chunk.tokens_count,
                            "title": chunk.metadata.title or "",
                            "author": chunk.metadata.author or "",
                            "source_url": chunk.metadata.source_url or "",
                            "file_path": chunk.metadata.file_path or "",
                            "document_type": chunk.metadata.document_type.value if chunk.metadata.document_type else "",
                            "created_at": chunk.metadata.created_at or ""
                        }
                        
                        # Add custom fields
                        data_object.update(chunk.metadata.custom_fields)
                        
                        batch.add_data_object(
                            data_object=data_object,
                            class_name=self.config.collection_name,
                            uuid=chunk.id,
                            vector=chunk.embedding
                        )
            
            await asyncio.get_event_loop().run_in_executor(None, insert_sync)
            
            VECTOR_OPERATIONS_COUNTER.labels(
                operation="insert",
                collection=self.config.collection_name,
                status="success"
            ).inc(len(chunks))
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert chunks into Weaviate: {e}")
            VECTOR_OPERATIONS_COUNTER.labels(
                operation="insert",
                collection=self.config.collection_name,
                status="error"
            ).inc(len(chunks))
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        collection_name: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search in Weaviate."""
        collection = collection_name or self.config.collection_name
        start_time = time.time()
        
        try:
            def search_sync():
                query_builder = self.client.query.get(
                    collection,
                    ["content", "chunk_index", "start_char", "end_char", "tokens_count",
                     "title", "author", "source_url", "file_path", "document_type", "created_at"]
                ).with_near_vector({
                    "vector": query_embedding
                }).with_limit(k).with_additional(["certainty"])
                
                # Add filters
                if filters:
                    where_filter = self._build_where_filter(filters)
                    if where_filter:
                        query_builder = query_builder.with_where(where_filter)
                
                return query_builder.do()
            
            results = await asyncio.get_event_loop().run_in_executor(None, search_sync)
            
            # Convert to SearchResult objects
            search_results = []
            if "data" in results and "Get" in results["data"] and collection in results["data"]["Get"]:
                for i, item in enumerate(results["data"]["Get"][collection]):
                    # Create metadata
                    metadata = DocumentMetadata(
                        title=item.get("title"),
                        author=item.get("author"),
                        source_url=item.get("source_url"),
                        file_path=item.get("file_path"),
                        document_type=DocumentType(item["document_type"]) if item.get("document_type") else None,
                        created_at=item.get("created_at")
                    )
                    
                    # Create chunk
                    chunk = DocumentChunk(
                        id=item.get("_additional", {}).get("id", ""),
                        content=item["content"],
                        metadata=metadata,
                        chunk_index=item.get("chunk_index", 0),
                        start_char=item.get("start_char", 0),
                        end_char=item.get("end_char", 0),
                        tokens_count=item.get("tokens_count", 0)
                    )
                    
                    # Get similarity score
                    certainty = item.get("_additional", {}).get("certainty", 0.0)
                    similarity_score = certainty  # Weaviate uses certainty (0-1)
                    
                    search_results.append(SearchResult(
                        chunk=chunk,
                        similarity_score=similarity_score,
                        rank=i + 1
                    ))
            
            VECTOR_SEARCH_LATENCY.labels(
                collection=collection,
                k=k
            ).observe(time.time() - start_time)
            
            VECTOR_OPERATIONS_COUNTER.labels(
                operation="search",
                collection=collection,
                status="success"
            ).inc()
            
            return search_results
        
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            VECTOR_OPERATIONS_COUNTER.labels(
                operation="search",
                collection=collection,
                status="error"
            ).inc()
            return []
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter from filters dict."""
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueText": value
                })
            elif isinstance(value, (int, float)):
                conditions.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueNumber": value
                })
            elif isinstance(value, list):
                # OR condition for multiple values
                or_conditions = []
                for v in value:
                    if isinstance(v, str):
                        or_conditions.append({
                            "path": [key],
                            "operator": "Equal",
                            "valueText": v
                        })
                if or_conditions:
                    conditions.append({
                        "operator": "Or",
                        "operands": or_conditions
                    })
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {
                "operator": "And",
                "operands": conditions
            }
        return None
    
    async def delete_by_id(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by IDs."""
        try:
            def delete_sync():
                for chunk_id in chunk_ids:
                    self.client.data_object.delete(
                        uuid=chunk_id,
                        class_name=self.config.collection_name
                    )
            
            await asyncio.get_event_loop().run_in_executor(None, delete_sync)
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete from Weaviate: {e}")
            return False
    
    async def update_chunk(self, chunk: DocumentChunk) -> bool:
        """Update a chunk in Weaviate."""
        try:
            data_object = {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "tokens_count": chunk.tokens_count,
                "title": chunk.metadata.title or "",
                "author": chunk.metadata.author or "",
                "source_url": chunk.metadata.source_url or "",
                "file_path": chunk.metadata.file_path or "",
                "document_type": chunk.metadata.document_type.value if chunk.metadata.document_type else "",
                "created_at": chunk.metadata.created_at or ""
            }
            
            def update_sync():
                self.client.data_object.replace(
                    uuid=chunk.id,
                    class_name=self.config.collection_name,
                    data_object=data_object,
                    vector=chunk.embedding
                )
            
            await asyncio.get_event_loop().run_in_executor(None, update_sync)
            return True
        
        except Exception as e:
            logger.error(f"Failed to update chunk in Weaviate: {e}")
            return False
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get Weaviate collection statistics."""
        try:
            def get_stats_sync():
                # Get object count
                result = self.client.query.aggregate(collection_name).with_meta_count().do()
                count = 0
                if "data" in result and "Aggregate" in result["data"]:
                    agg_data = result["data"]["Aggregate"].get(collection_name)
                    if agg_data and len(agg_data) > 0:
                        count = agg_data[0].get("meta", {}).get("count", 0)
                
                return {"document_count": count}
            
            stats = await asyncio.get_event_loop().run_in_executor(None, get_stats_sync)
            
            VECTOR_DB_SIZE_GAUGE.labels(collection=collection_name).set(stats["document_count"])
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get Weaviate stats: {e}")
            return {"document_count": 0}
    
    async def close(self) -> None:
        """Close Weaviate client."""
        if self.client:
            # Weaviate client doesn't have explicit close method
            self.client = None


class VectorService:
    """Advanced vector database service with RAG capabilities."""
    
    def __init__(
        self,
        vector_config: VectorDBConfig,
        embedding_config: EmbeddingConfig,
        chunking_config: ChunkingConfig
    ):
        self.vector_config = vector_config
        self.embedding_config = embedding_config
        self.chunking_config = chunking_config
        
        self.vector_db = self._create_vector_db()
        self.embedding_model = EmbeddingModel(embedding_config)
        self.document_processor = DocumentProcessor(chunking_config)
        
        self._initialized = False
    
    def _create_vector_db(self) -> BaseVectorDB:
        """Create vector database instance based on provider."""
        if self.vector_config.provider == VectorDBProvider.WEAVIATE:
            return WeaviateVectorDB(self.vector_config)
        elif self.vector_config.provider == VectorDBProvider.MILVUS:
            # Would implement MilvusVectorDB here
            raise NotImplementedError("Milvus support not yet implemented")
        else:
            raise ValueError(f"Unsupported vector DB provider: {self.vector_config.provider}")
    
    async def initialize(self) -> None:
        """Initialize the vector service."""
        if self._initialized:
            return
        
        await self.vector_db.initialize()
        await self.embedding_model.initialize()
        
        # Create default collection if it doesn't exist
        await self.vector_db.create_collection(
            self.vector_config.collection_name,
            self.embedding_config.dimensions
        )
        
        self._initialized = True
        logger.info("Vector service initialized")
    
    async def ingest_document(
        self,
        content: Union[str, bytes],
        document_type: DocumentType,
        metadata: Optional[DocumentMetadata] = None
    ) -> Dict[str, Any]:
        """Ingest a document into the vector database."""
        start_time = time.time()
        
        try:
            # Process document into chunks
            chunks = await self.document_processor.process_document(
                content, document_type, metadata
            )
            
            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_model.encode(texts)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Insert into vector database
            success = await self.vector_db.insert_chunks(chunks)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = {
                "success": success,
                "chunks_created": len(chunks),
                "processing_time_ms": processing_time_ms,
                "document_metadata": metadata.__dict__ if metadata else None
            }
            
            if success:
                logger.info(
                    f"Ingested document: {len(chunks)} chunks, "
                    f"{processing_time_ms:.2f}ms"
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise
    
    async def ingest_file(self, file_path: Path) -> Dict[str, Any]:
        """Ingest a file into the vector database."""
        chunks = await self.document_processor.process_file(file_path)
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_model.encode(texts)
        
        # Assign embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Insert into vector database
        success = await self.vector_db.insert_chunks(chunks)
        
        return {
            "success": success,
            "chunks_created": len(chunks),
            "file_path": str(file_path)
        }
    
    async def search(
        self,
        query: str,
        k: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Search for relevant document chunks."""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.encode_query(query)
            
            # Search in vector database
            results = await self.vector_db.search(
                query_embedding=query_embedding,
                collection_name=collection_name,
                k=k,
                filters=filters
            )
            
            # Optional re-ranking (could implement more sophisticated ranking here)
            if rerank:
                results = await self._rerank_results(query, results)
            
            search_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Search completed: {len(results)} results, "
                f"{search_time_ms:.2f}ms"
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank search results (placeholder for advanced re-ranking)."""
        # Simple re-ranking based on content length and similarity
        # In practice, you might use cross-encoders or other sophisticated methods
        
        for result in results:
            # Adjust score based on content quality indicators
            content_length_factor = min(len(result.chunk.content) / 500, 1.0)  # Prefer longer chunks
            result.similarity_score *= (0.8 + 0.2 * content_length_factor)
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    async def get_similar_chunks(
        self,
        chunk_id: str,
        k: int = 5
    ) -> List[SearchResult]:
        """Get chunks similar to a given chunk."""
        # This would require fetching the chunk's embedding and searching
        # Implementation depends on the vector database's capabilities
        raise NotImplementedError("Similar chunks search not yet implemented")
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks of a document."""
        # This would require tracking document-to-chunks mapping
        # Implementation depends on metadata structure
        raise NotImplementedError("Document deletion not yet implemented")
    
    async def update_chunk(self, chunk: DocumentChunk) -> bool:
        """Update a document chunk."""
        # Re-generate embedding if content changed
        if not chunk.embedding:
            chunk.embedding = await self.embedding_model.encode_query(chunk.content)
        
        return await self.vector_db.update_chunk(chunk)
    
    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get vector database statistics."""
        collection = collection_name or self.vector_config.collection_name
        return await self.vector_db.get_collection_stats(collection)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of vector service components."""
        health = {
            "embedding_model": self.embedding_model._model_loaded,
            "vector_db": self._initialized
        }
        
        try:
            # Test vector DB with a simple operation
            stats = await self.get_collection_stats()
            health["vector_db"] = True
        except Exception:
            health["vector_db"] = False
        
        health["overall"] = all(health.values())
        
        return health
    
    async def close(self) -> None:
        """Close the vector service."""
        await self.vector_db.close()
        logger.info("Vector service closed")


# Utility functions for RAG
async def create_rag_context(
    search_results: List[SearchResult],
    max_context_length: int = 4000,
    include_metadata: bool = True
) -> str:
    """Create context string from search results for RAG."""
    context_parts = []
    current_length = 0
    
    for result in search_results:
        chunk = result.chunk
        
        # Format chunk content
        chunk_text = f"[Source: {chunk.metadata.title or 'Unknown'}]\n{chunk.content}\n"
        
        if include_metadata and chunk.metadata.source_url:
            chunk_text += f"URL: {chunk.metadata.source_url}\n"
        
        chunk_text += "---\n"
        
        # Check if adding this chunk exceeds max length
        if current_length + len(chunk_text) > max_context_length:
            break
        
        context_parts.append(chunk_text)
        current_length += len(chunk_text)
    
    return "\n".join(context_parts)


def create_rag_prompt(query: str, context: str, system_prompt: Optional[str] = None) -> str:
    """Create a RAG prompt combining query and context."""
    if system_prompt:
        prompt = f"{system_prompt}\n\n"
    else:
        prompt = "You are a helpful assistant. Use the provided context to answer the question. If the context doesn't contain relevant information, say so.\n\n"
    
    prompt += f"Context:\n{context}\n\n"
    prompt += f"Question: {query}\n\n"
    prompt += "Answer:"
    
    return prompt