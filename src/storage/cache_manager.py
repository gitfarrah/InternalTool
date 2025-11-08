"""
Cache Manager for search results and API responses.

Provides in-memory caching with TTL support to reduce API calls
and improve response times for repeated queries.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple in-memory cache for search results and API responses."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        """
        Initialize cache manager.
        
        Args:
            default_ttl: Default time-to-live in seconds for cached items
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_key(self, query: str, filters: Dict[str, Any]) -> str:
        """
        Generate a cache key from query and filters.
        
        Args:
            query: Search query string
            filters: Dictionary of filter parameters
            
        Returns:
            MD5 hash of the normalized query and filters
        """
        # Create a deterministic key from query and filters
        key_data = {
            "query": query.strip().lower(),
            "filters": {k: v for k, v in filters.items() if v is not None}
        }
        # Sort items for consistent key generation
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available and not expired.
        
        Args:
            query: Search query string
            filters: Dictionary of filter parameters
            
        Returns:
            Cached data if found and not expired, None otherwise
        """
        cache_key = self._generate_key(query, filters)
        
        if cache_key not in self._cache:
            return None
        
        cached_item = self._cache[cache_key]
        
        # Check if expired
        if time.time() > cached_item["expires_at"]:
            del self._cache[cache_key]
            return None
        
        logger.info(f"Cache hit for query: {query[:50]}...")
        return cached_item["data"]
    
    def set(self, query: str, filters: Dict[str, Any], data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Cache the result with TTL.
        
        Args:
            query: Search query string
            filters: Dictionary of filter parameters
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if not provided)
        """
        cache_key = self._generate_key(query, filters)
        expires_at = time.time() + (ttl or self.default_ttl)
        
        self._cache[cache_key] = {
            "data": data,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        
        logger.info(f"Cached result for query: {query[:50]}... (expires in {ttl or self.default_ttl}s)")
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self._cache.items()
            if current_time > item["expires_at"]
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics (total_items, active_items, expired_items, memory_usage)
        """
        current_time = time.time()
        active_items = sum(1 for item in self._cache.values() if current_time <= item["expires_at"])
        expired_items = len(self._cache) - active_items
        
        return {
            "total_items": len(self._cache),
            "active_items": active_items,
            "expired_items": expired_items,
            "memory_usage": len(str(self._cache))
        }


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        Global CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_intent_analysis(query: str, filters: Dict[str, Any], intent_data: Dict[str, Any]) -> None:
    """
    Cache intent analysis results.
    
    Args:
        query: Search query string
        filters: Dictionary of filter parameters
        intent_data: Intent analysis result to cache
    """
    cache_manager = get_cache_manager()
    cache_manager.set(query, filters, intent_data, ttl=600)  # 10 minutes for intent analysis


def get_cached_intent_analysis(query: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get cached intent analysis results.
    
    Args:
        query: Search query string
        filters: Dictionary of filter parameters
        
    Returns:
        Cached intent analysis if found, None otherwise
    """
    cache_manager = get_cache_manager()
    return cache_manager.get(query, filters)


def cache_search_results(query: str, filters: Dict[str, Any], results: Dict[str, Any]) -> None:
    """
    Cache search results.
    
    Args:
        query: Search query string
        filters: Dictionary of filter parameters
        results: Search results to cache
    """
    cache_manager = get_cache_manager()
    # Cache for 1 hour for search results (3600 seconds)
    cache_manager.set(query, filters, results, ttl=3600)


def get_cached_search_results(query: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get cached search results.
    
    Args:
        query: Search query string
        filters: Dictionary of filter parameters
        
    Returns:
        Cached search results if found, None otherwise
    """
    cache_manager = get_cache_manager()
    return cache_manager.get(query, filters)


def cache_llm_response(query: str, context: str, response: Dict[str, Any]) -> None:
    """
    Cache LLM (Gemini) responses.
    
    Args:
        query: User query string
        context: Context passed to LLM
        response: LLM response to cache
    """
    cache_manager = get_cache_manager()
    # Use query + context hash as filters for uniqueness
    filters = {"context_hash": hashlib.md5(context.encode()).hexdigest()}
    # Cache for 30 minutes for LLM responses
    cache_manager.set(query, filters, response, ttl=1800)


def get_cached_llm_response(query: str, context: str) -> Optional[Dict[str, Any]]:
    """
    Get cached LLM response.
    
    Args:
        query: User query string
        context: Context passed to LLM
        
    Returns:
        Cached LLM response if found, None otherwise
    """
    cache_manager = get_cache_manager()
    filters = {"context_hash": hashlib.md5(context.encode()).hexdigest()}
    return cache_manager.get(query, filters)


__all__ = [
    "CacheManager",
    "get_cache_manager",
    "cache_intent_analysis",
    "get_cached_intent_analysis",
    "cache_search_results",
    "get_cached_search_results",
    "cache_llm_response",
    "get_cached_llm_response",
]

