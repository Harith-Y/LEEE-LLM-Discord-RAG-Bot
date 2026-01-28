"""
Caching layer for index and response caching
"""
import asyncio
import hashlib
import time
from typing import Optional, Any, Dict, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Thread-safe Least Recently Used (LRU) cache implementation
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str, ttl: Optional[float] = None) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds (None = no expiry check)
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self.cache:
                return None
            
            value, timestamp = self.cache[key]
            
            # Check if expired
            if ttl is not None and (time.time() - timestamp) > ttl:
                del self.cache[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return value
    
    async def set(self, key: str, value: Any) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # If key exists, remove it first
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = (value, time.time())
            
            # Evict oldest if over max size
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:50]}...")
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
        logger.info("Cache cleared")
    
    async def size(self) -> int:
        """Get current cache size"""
        async with self._lock:
            return len(self.cache)
    
    async def remove(self, key: str) -> bool:
        """
        Remove specific key from cache
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if key was removed, False if not found
        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False


class CacheManager:
    """
    Manages different cache types for the application
    """
    
    def __init__(self, response_cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            response_cache_size: Max size for response cache
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.response_cache = LRUCache(max_size=response_cache_size)
        self.cache_ttl = cache_ttl
        self._index_cache: Optional[Any] = None
        self._index_lock = asyncio.Lock()
        logger.info(f"CacheManager initialized - Response cache size: {response_cache_size}, TTL: {cache_ttl}s")
    
    async def get_cached_response(self, query: str) -> Optional[str]:
        """
        Get cached response for a query
        
        Args:
            query: User query
            
        Returns:
            Cached response or None
        """
        cache_key = self._generate_cache_key(query)
        response = await self.response_cache.get(cache_key, ttl=self.cache_ttl)
        
        if response:
            logger.debug(f"Cache HIT for query: {query[:50]}...")
        else:
            logger.debug(f"Cache MISS for query: {query[:50]}...")
        
        return response
    
    async def set_cached_response(self, query: str, response: str) -> None:
        """
        Cache a response for a query
        
        Args:
            query: User query
            response: Bot response
        """
        cache_key = self._generate_cache_key(query)
        await self.response_cache.set(cache_key, response)
        logger.debug(f"Cached response for query: {query[:50]}...")
    
    async def get_cached_index(self) -> Optional[Any]:
        """
        Get cached index
        
        Returns:
            Cached index or None
        """
        async with self._index_lock:
            return self._index_cache
    
    async def set_cached_index(self, index: Any) -> None:
        """
        Cache the index
        
        Args:
            index: Index to cache
        """
        async with self._index_lock:
            self._index_cache = index
        logger.info("Index cached successfully")
    
    async def invalidate_index_cache(self) -> None:
        """Invalidate the cached index (e.g., after database update)"""
        async with self._index_lock:
            self._index_cache = None
        logger.info("Index cache invalidated")
    
    async def clear_all_caches(self) -> None:
        """Clear all caches"""
        await self.response_cache.clear()
        await self.invalidate_index_cache()
        logger.info("All caches cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        response_cache_size = await self.response_cache.size()
        index_cached = self._index_cache is not None
        
        return {
            'response_cache_size': response_cache_size,
            'response_cache_max_size': self.response_cache.max_size,
            'index_cached': index_cached,
            'cache_ttl_seconds': self.cache_ttl,
        }
    
    @staticmethod
    def _generate_cache_key(query: str) -> str:
        """
        Generate cache key from query
        
        Args:
            query: User query
            
        Returns:
            MD5 hash of normalized query
        """
        # Normalize query: lowercase and strip whitespace
        normalized = query.lower().strip()
        # Generate hash
        return hashlib.md5(normalized.encode()).hexdigest()


# Global cache manager instance
cache_manager: Optional[CacheManager] = None


def get_cache_manager(response_cache_size: int = 1000, cache_ttl: int = 3600) -> CacheManager:
    """
    Get or create global cache manager instance
    
    Args:
        response_cache_size: Max size for response cache
        cache_ttl: Time-to-live for cache entries in seconds
        
    Returns:
        CacheManager instance
    """
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager(response_cache_size, cache_ttl)
    return cache_manager
