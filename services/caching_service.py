"""
Advanced Caching Service for Enterprise AI Gateway

Provides multi-layer caching with Redis clustering, intelligent cache warming,
performance optimization, and comprehensive monitoring.
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from collections import defaultdict, OrderedDict
import threading

import aioredis
import structlog
from prometheus_client import Counter, Histogram, Gauge
from redis.sentinel import Sentinel

logger = structlog.get_logger(__name__)

# Metrics
CACHE_OPERATIONS_COUNTER = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'cache_type', 'status']
)

CACHE_HIT_RATIO_GAUGE = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type']
)

CACHE_LATENCY_HISTOGRAM = Histogram(
    'cache_operation_duration_seconds',
    'Cache operation duration',
    ['operation', 'cache_type']
)

CACHE_SIZE_GAUGE = Gauge(
    'cache_size_bytes',
    'Cache size in bytes',
    ['cache_type']
)

CACHE_EVICTIONS_COUNTER = Counter(
    'cache_evictions_total',
    'Total cache evictions',
    ['cache_type', 'reason']
)


class CacheStrategy(str, Enum):
    """Cache strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"


class CacheLevel(str, Enum):
    """Cache levels."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_PERSISTENT = "l3_persistent"


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    strategy: CacheStrategy = CacheStrategy.LRU
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    max_entries: int = 10000
    eviction_policy: str = "lru"
    compression: bool = True
    serialization: str = "pickle"  # pickle, json, msgpack
    
    # Redis specific
    redis_url: str = "redis://localhost:6379/0"
    redis_cluster_nodes: List[str] = field(default_factory=list)
    redis_sentinel_hosts: List[Tuple[str, int]] = field(default_factory=list)
    redis_sentinel_service: str = "mymaster"
    redis_pool_size: int = 10
    redis_retry_attempts: int = 3
    
    # Performance settings
    batch_size: int = 100
    pipeline_enabled: bool = True
    async_write_enabled: bool = True
    
    # Monitoring
    metrics_enabled: bool = True
    statistics_interval: int = 60


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[int] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Cache statistics collector."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_size_bytes = 0
        self.entry_count = 0
        self._lock = threading.Lock()
    
    def record_hit(self) -> None:
        with self._lock:
            self.hits += 1
    
    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1
    
    def record_set(self, size_bytes: int = 0) -> None:
        with self._lock:
            self.sets += 1
            self.total_size_bytes += size_bytes
            self.entry_count += 1
    
    def record_delete(self, size_bytes: int = 0) -> None:
        with self._lock:
            self.deletes += 1
            self.total_size_bytes -= size_bytes
            self.entry_count -= 1
    
    def record_eviction(self, size_bytes: int = 0) -> None:
        with self._lock:
            self.evictions += 1
            self.total_size_bytes -= size_bytes
            self.entry_count -= 1
    
    def record_error(self) -> None:
        with self._lock:
            self.errors += 1
    
    @property
    def hit_ratio(self) -> float:
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "evictions": self.evictions,
                "errors": self.errors,
                "hit_ratio": self.hit_ratio,
                "total_size_bytes": self.total_size_bytes,
                "entry_count": self.entry_count
            }


class CacheSerializer:
    """Cache serialization handler."""
    
    @staticmethod
    def serialize(value: Any, method: str = "pickle") -> bytes:
        """Serialize value to bytes."""
        if method == "pickle":
            return pickle.dumps(value)
        elif method == "json":
            return json.dumps(value, ensure_ascii=False).encode("utf-8")
        else:
            raise ValueError(f"Unsupported serialization method: {method}")
    
    @staticmethod
    def deserialize(data: bytes, method: str = "pickle") -> Any:
        """Deserialize bytes to value."""
        if method == "pickle":
            return pickle.loads(data)
        elif method == "json":
            return json.loads(data.decode("utf-8"))
        else:
            raise ValueError(f"Unsupported serialization method: {method}")


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    def __init__(self, config: CacheConfig, cache_level: CacheLevel):
        self.config = config
        self.cache_level = cache_level
        self.stats = CacheStats()
        self._initialized = False
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get cache size in bytes."""
        pass
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple values in cache."""
        success_count = 0
        for key, value in items.items():
            if await self.set(key, value, ttl):
                success_count += 1
        return success_count
    
    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from cache."""
        success_count = 0
        for key in keys:
            if await self.delete(key):
                success_count += 1
        return success_count
    
    async def initialize(self) -> None:
        """Initialize the cache."""
        self._initialized = True
        logger.info(f"Initialized {self.cache_level} cache")
    
    async def close(self) -> None:
        """Close the cache."""
        logger.info(f"Closed {self.cache_level} cache")


class MemoryCache(BaseCache):
    """In-memory LRU cache implementation."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config, CacheLevel.L1_MEMORY)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._background_task = None
    
    async def initialize(self) -> None:
        await super().initialize()
        if self.config.metrics_enabled:
            self._background_task = asyncio.create_task(self._metrics_reporter())
    
    async def get(self, key: str) -> Optional[Any]:
        start_time = time.time()
        
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.stats.record_miss()
                CACHE_OPERATIONS_COUNTER.labels(
                    operation="get",
                    cache_type="memory",
                    status="miss"
                ).inc()
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self.stats.record_miss()
                self.stats.record_eviction(entry.size_bytes)
                CACHE_EVICTIONS_COUNTER.labels(
                    cache_type="memory",
                    reason="expired"
                ).inc()
                return None
            
            # Move to end (LRU)
            entry.touch()
            self._cache.move_to_end(key)
            
            self.stats.record_hit()
            CACHE_OPERATIONS_COUNTER.labels(
                operation="get",
                cache_type="memory",
                status="hit"
            ).inc()
            
            CACHE_LATENCY_HISTOGRAM.labels(
                operation="get",
                cache_type="memory"
            ).observe(time.time() - start_time)
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        start_time = time.time()
        
        try:
            # Serialize to calculate size
            serialized = CacheSerializer.serialize(value, self.config.serialization)
            size_bytes = len(serialized)
            
            # Check size limits
            if size_bytes > self.config.max_size_mb * 1024 * 1024:
                logger.warning(f"Entry too large for cache: {size_bytes} bytes")
                return False
            
            async with self._lock:
                # Remove existing entry
                if key in self._cache:
                    old_entry = self._cache[key]
                    self.stats.record_delete(old_entry.size_bytes)
                
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl=ttl or self.config.ttl_seconds,
                    size_bytes=size_bytes
                )
                
                self._cache[key] = entry
                self._cache.move_to_end(key)
                
                # Evict if necessary
                await self._evict_if_needed()
                
                self.stats.record_set(size_bytes)
                CACHE_OPERATIONS_COUNTER.labels(
                    operation="set",
                    cache_type="memory",
                    status="success"
                ).inc()
                
                CACHE_LATENCY_HISTOGRAM.labels(
                    operation="set",
                    cache_type="memory"
                ).observe(time.time() - start_time)
                
                return True
        
        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
            self.stats.record_error()
            CACHE_OPERATIONS_COUNTER.labels(
                operation="set",
                cache_type="memory",
                status="error"
            ).inc()
            return False
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self.stats.record_delete(entry.size_bytes)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        async with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired
    
    async def clear(self) -> bool:
        async with self._lock:
            self._cache.clear()
            self.stats = CacheStats()  # Reset stats
            return True
    
    async def size(self) -> int:
        return self.stats.total_size_bytes
    
    async def _evict_if_needed(self) -> None:
        """Evict entries if cache is over limits."""
        while (
            len(self._cache) > self.config.max_entries or
            self.stats.total_size_bytes > self.config.max_size_mb * 1024 * 1024
        ):
            if not self._cache:
                break
            
            # Remove oldest entry (FIFO/LRU)
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache.pop(oldest_key)
            
            self.stats.record_eviction(oldest_entry.size_bytes)
            CACHE_EVICTIONS_COUNTER.labels(
                cache_type="memory",
                reason="size_limit"
            ).inc()
    
    async def _metrics_reporter(self) -> None:
        """Background task to report metrics."""
        while True:
            try:
                stats = self.stats.get_stats()
                CACHE_HIT_RATIO_GAUGE.labels(cache_type="memory").set(stats["hit_ratio"])
                CACHE_SIZE_GAUGE.labels(cache_type="memory").set(stats["total_size_bytes"])
                
                await asyncio.sleep(self.config.statistics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics reporting error: {e}")
                await asyncio.sleep(5)
    
    async def close(self) -> None:
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        await super().close()


class RedisCache(BaseCache):
    """Redis-based distributed cache."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config, CacheLevel.L2_REDIS)
        self.redis_client = None
        self._connection_pool = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            if self.config.redis_cluster_nodes:
                # Redis Cluster
                from aioredis.cluster import RedisCluster
                self.redis_client = RedisCluster(
                    startup_nodes=self.config.redis_cluster_nodes,
                    decode_responses=False,
                    skip_full_coverage_check=True
                )
            elif self.config.redis_sentinel_hosts:
                # Redis Sentinel
                sentinel = Sentinel(self.config.redis_sentinel_hosts)
                master = sentinel.master_for(
                    self.config.redis_sentinel_service,
                    decode_responses=False
                )
                self.redis_client = aioredis.Redis(connection_pool=master.connection_pool)
            else:
                # Single Redis instance
                self.redis_client = aioredis.from_url(
                    self.config.redis_url,
                    decode_responses=False,
                    max_connections=self.config.redis_pool_size,
                    retry_on_timeout=True
                )
            
            # Test connection
            await self.redis_client.ping()
            await super().initialize()
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        
        start_time = time.time()
        
        try:
            data = await self.redis_client.get(key)
            
            if data is None:
                self.stats.record_miss()
                CACHE_OPERATIONS_COUNTER.labels(
                    operation="get",
                    cache_type="redis",
                    status="miss"
                ).inc()
                return None
            
            value = CacheSerializer.deserialize(data, self.config.serialization)
            
            self.stats.record_hit()
            CACHE_OPERATIONS_COUNTER.labels(
                operation="get",
                cache_type="redis",
                status="hit"
            ).inc()
            
            CACHE_LATENCY_HISTOGRAM.labels(
                operation="get",
                cache_type="redis"
            ).observe(time.time() - start_time)
            
            return value
        
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.record_error()
            CACHE_OPERATIONS_COUNTER.labels(
                operation="get",
                cache_type="redis",
                status="error"
            ).inc()
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.redis_client:
            return False
        
        start_time = time.time()
        
        try:
            data = CacheSerializer.serialize(value, self.config.serialization)
            ttl_seconds = ttl or self.config.ttl_seconds
            
            result = await self.redis_client.setex(key, ttl_seconds, data)
            
            if result:
                self.stats.record_set(len(data))
                CACHE_OPERATIONS_COUNTER.labels(
                    operation="set",
                    cache_type="redis",
                    status="success"
                ).inc()
            
            CACHE_LATENCY_HISTOGRAM.labels(
                operation="set",
                cache_type="redis"
            ).observe(time.time() - start_time)
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.stats.record_error()
            CACHE_OPERATIONS_COUNTER.labels(
                operation="set",
                cache_type="redis",
                status="error"
            ).inc()
            return False
    
    async def delete(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            if result:
                self.stats.record_delete()
            return bool(result)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self.stats.record_error()
            return False
    
    async def exists(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushdb()
            self.stats = CacheStats()  # Reset stats
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def size(self) -> int:
        if not self.redis_client:
            return 0
        
        try:
            return await self.redis_client.dbsize()
        except Exception as e:
            logger.error(f"Redis size error: {e}")
            return 0
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        if not self.redis_client or not keys:
            return {}
        
        try:
            values = await self.redis_client.mget(keys)
            results = {}
            
            for key, data in zip(keys, values):
                if data is not None:
                    try:
                        value = CacheSerializer.deserialize(data, self.config.serialization)
                        results[key] = value
                        self.stats.record_hit()
                    except Exception as e:
                        logger.warning(f"Failed to deserialize key {key}: {e}")
                        self.stats.record_error()
                else:
                    self.stats.record_miss()
            
            return results
        
        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            self.stats.record_error()
            return {}
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> int:
        if not self.redis_client or not items:
            return 0
        
        try:
            pipe = self.redis_client.pipeline()
            ttl_seconds = ttl or self.config.ttl_seconds
            
            for key, value in items.items():
                data = CacheSerializer.serialize(value, self.config.serialization)
                pipe.setex(key, ttl_seconds, data)
            
            results = await pipe.execute()
            success_count = sum(1 for r in results if r)
            
            for _ in range(success_count):
                self.stats.record_set()
            
            return success_count
        
        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            self.stats.record_error()
            return 0
    
    async def close(self) -> None:
        if self.redis_client:
            await self.redis_client.close()
        await super().close()


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) layers."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.l1_cache = MemoryCache(config)
        self.l2_cache = RedisCache(config)
        self._write_behind_queue = asyncio.Queue()
        self._write_behind_task = None
    
    async def initialize(self) -> None:
        """Initialize all cache levels."""
        await self.l1_cache.initialize()
        
        try:
            await self.l2_cache.initialize()
        except Exception as e:
            logger.warning(f"L2 cache initialization failed, continuing with L1 only: {e}")
        
        if self.config.async_write_enabled:
            self._write_behind_task = asyncio.create_task(self._write_behind_worker())
        
        logger.info("Multi-level cache initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with L1 -> L2 fallback."""
        
        # Try L1 first
        value = await self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = await self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            await self.l1_cache.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both cache levels."""
        
        # Always set in L1
        l1_success = await self.l1_cache.set(key, value, ttl)
        
        # Set in L2 (async or sync based on config)
        if self.config.async_write_enabled:
            await self._write_behind_queue.put((key, value, ttl))
            return l1_success
        else:
            l2_success = await self.l2_cache.set(key, value, ttl)
            return l1_success and l2_success
    
    async def delete(self, key: str) -> bool:
        """Delete from both cache levels."""
        l1_result = await self.l1_cache.delete(key)
        l2_result = await self.l2_cache.delete(key)
        return l1_result or l2_result
    
    async def exists(self, key: str) -> bool:
        """Check existence in either cache level."""
        return await self.l1_cache.exists(key) or await self.l2_cache.exists(key)
    
    async def clear(self) -> bool:
        """Clear both cache levels."""
        l1_result = await self.l1_cache.clear()
        l2_result = await self.l2_cache.clear()
        return l1_result and l2_result
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        l1_stats = self.l1_cache.stats.get_stats()
        l2_stats = self.l2_cache.stats.get_stats()
        
        return {
            "l1_memory": l1_stats,
            "l2_redis": l2_stats,
            "combined": {
                "total_hits": l1_stats["hits"] + l2_stats["hits"],
                "total_misses": l1_stats["misses"] + l2_stats["misses"],
                "combined_hit_ratio": (l1_stats["hits"] + l2_stats["hits"]) / 
                    max(1, l1_stats["hits"] + l1_stats["misses"] + l2_stats["hits"] + l2_stats["misses"])
            }
        }
    
    async def _write_behind_worker(self) -> None:
        """Background worker for async writes to L2."""
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                # Get item with timeout to enable periodic flushing
                try:
                    item = await asyncio.wait_for(
                        self._write_behind_queue.get(),
                        timeout=1.0
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # Flush if batch is full or timeout reached
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (batch and time.time() - last_flush > 5.0)
                )
                
                if should_flush and batch:
                    await self._flush_write_behind_batch(batch)
                    batch.clear()
                    last_flush = time.time()
                    
            except asyncio.CancelledError:
                # Flush remaining items before shutting down
                if batch:
                    await self._flush_write_behind_batch(batch)
                break
            except Exception as e:
                logger.error(f"Write-behind worker error: {e}")
                await asyncio.sleep(1)
    
    async def _flush_write_behind_batch(self, batch: List[Tuple]) -> None:
        """Flush a batch of writes to L2."""
        if not batch:
            return
        
        items = {}
        ttl = None
        
        for key, value, item_ttl in batch:
            items[key] = value
            if ttl is None:
                ttl = item_ttl
        
        try:
            await self.l2_cache.set_many(items, ttl)
        except Exception as e:
            logger.error(f"Write-behind batch flush error: {e}")
    
    async def close(self) -> None:
        """Close all cache levels."""
        if self._write_behind_task:
            self._write_behind_task.cancel()
            try:
                await self._write_behind_task
            except asyncio.CancelledError:
                pass
        
        await self.l1_cache.close()
        await self.l2_cache.close()
        logger.info("Multi-level cache closed")


class CacheWarmer:
    """Intelligent cache warming service."""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._warming_tasks: Set[str] = set()
        self._prediction_model = None  # Could integrate ML model for better predictions
    
    def record_access(self, key: str) -> None:
        """Record cache access for pattern analysis."""
        self._access_patterns[key].append(time.time())
        
        # Keep only recent access times (last 24 hours)
        cutoff = time.time() - 24 * 3600
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > cutoff
        ]
    
    async def predict_hot_keys(self, limit: int = 100) -> List[str]:
        """Predict keys that are likely to be accessed soon."""
        key_scores = {}
        current_time = time.time()
        
        for key, access_times in self._access_patterns.items():
            if not access_times:
                continue
            
            # Calculate access frequency (accesses per hour)
            recent_accesses = [t for t in access_times if t > current_time - 3600]
            frequency = len(recent_accesses)
            
            # Calculate recency score (more recent = higher score)
            if access_times:
                last_access = max(access_times)
                recency = 1 / (current_time - last_access + 1)
            else:
                recency = 0
            
            # Combined score
            key_scores[key] = frequency * recency
        
        # Return top keys sorted by score
        hot_keys = sorted(key_scores.keys(), key=lambda k: key_scores[k], reverse=True)
        return hot_keys[:limit]
    
    async def warm_cache(self, key_value_pairs: Dict[str, Any]) -> int:
        """Warm cache with provided key-value pairs."""
        warmed_count = 0
        
        for key, value in key_value_pairs.items():
            if key not in self._warming_tasks:
                self._warming_tasks.add(key)
                try:
                    success = await self.cache.set(key, value)
                    if success:
                        warmed_count += 1
                finally:
                    self._warming_tasks.discard(key)
        
        return warmed_count
    
    async def warm_from_source(
        self,
        keys: List[str],
        source_func: Callable[[str], Any]
    ) -> int:
        """Warm cache by fetching data from source function."""
        warmed_count = 0
        
        for key in keys:
            if key not in self._warming_tasks:
                self._warming_tasks.add(key)
                try:
                    if not await self.cache.exists(key):
                        value = await source_func(key)
                        if value is not None:
                            success = await self.cache.set(key, value)
                            if success:
                                warmed_count += 1
                finally:
                    self._warming_tasks.discard(key)
        
        return warmed_count


class CachingService:
    """Advanced caching service with multi-level caching and intelligent warming."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = MultiLevelCache(config)
        self.warmer = CacheWarmer(self.cache)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the caching service."""
        if self._initialized:
            return
        
        await self.cache.initialize()
        self._initialized = True
        logger.info("Caching service initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access pattern recording."""
        value = await self.cache.get(key)
        
        # Record access for warming patterns
        self.warmer.record_access(key)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return await self.cache.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.cache.exists(key)
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        return await self.cache.clear()
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        # Record access patterns
        for key in keys:
            self.warmer.record_access(key)
        
        # Try L1 first for all keys
        l1_results = await self.cache.l1_cache.get_many(keys)
        
        # Find missing keys
        missing_keys = [k for k in keys if k not in l1_results]
        
        if missing_keys:
            # Get missing keys from L2
            l2_results = await self.cache.l2_cache.get_many(missing_keys)
            
            # Promote L2 hits to L1
            if l2_results:
                await self.cache.l1_cache.set_many(l2_results)
            
            # Combine results
            l1_results.update(l2_results)
        
        return l1_results
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple values in cache."""
        return await self.cache.l1_cache.set_many(items, ttl)
    
    async def warm_cache(self, key_value_pairs: Dict[str, Any]) -> int:
        """Warm cache with provided data."""
        return await self.warmer.warm_cache(key_value_pairs)
    
    async def predict_and_warm(self, source_func: Callable[[str], Any], limit: int = 50) -> int:
        """Predict hot keys and warm cache proactively."""
        hot_keys = await self.warmer.predict_hot_keys(limit)
        return await self.warmer.warm_from_source(hot_keys, source_func)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_stats = await self.cache.get_stats()
        
        # Add service-level stats
        service_stats = {
            "cache_levels": cache_stats,
            "warming_tasks_active": len(self.warmer._warming_tasks),
            "access_patterns_tracked": len(self.warmer._access_patterns),
            "config": {
                "strategy": self.config.strategy,
                "ttl_seconds": self.config.ttl_seconds,
                "max_size_mb": self.config.max_size_mb,
                "max_entries": self.config.max_entries
            }
        }
        
        return service_stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all cache components."""
        health = {}
        
        # Check L1 (always healthy if initialized)
        health["l1_memory"] = self._initialized
        
        # Check L2 Redis
        try:
            if hasattr(self.cache.l2_cache, 'redis_client') and self.cache.l2_cache.redis_client:
                await self.cache.l2_cache.redis_client.ping()
                health["l2_redis"] = True
            else:
                health["l2_redis"] = False
        except Exception:
            health["l2_redis"] = False
        
        health["overall"] = health["l1_memory"] and health.get("l2_redis", False)
        
        return health
    
    async def close(self) -> None:
        """Close the caching service."""
        await self.cache.close()
        logger.info("Caching service closed")


# Utility functions
def create_cache_key(*args, separator: str = ":") -> str:
    """Create a standardized cache key from arguments."""
    key_parts = []
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:8])
        else:
            key_parts.append(str(arg))
    return separator.join(key_parts)


def cache_key_for_llm_request(message: str, model: str, temperature: float, **kwargs) -> str:
    """Generate cache key for LLM requests."""
    key_data = {
        "message": message,
        "model": model,
        "temperature": temperature,
        **kwargs
    }
    return f"llm:{hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()}"