#!/usr/bin/env python3
"""
HTML Cache Service - Manages HTML content caching with access counters
"""

import logging
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List
from ..models.cache_models import CachedContent, CacheStatsResponse, CacheCleanupResponse
from .mongodb_service import MongoDBService

logger = logging.getLogger(__name__)


class HTMLCacheService:
    """Service for managing HTML content caching with access counting"""
    
    def __init__(self, mongodb_service: MongoDBService, default_expiry_days: int = 30):
        self.mongodb_service = mongodb_service
        self.default_expiry_days = default_expiry_days
        self._initialized = False
        
        # Cache performance metrics
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"HTMLCacheService initialized with {default_expiry_days} day default expiry")
    
    async def initialize(self) -> None:
        """Initialize the cache service and ensure indexes exist"""
        if not self._initialized:
            try:
                # Ensure MongoDB client is initialized
                await self.mongodb_service.create_indexes()
                self._initialized = True
                logger.info("HTMLCacheService initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize HTMLCacheService: {e}")
                raise
    
    async def get_or_fetch_content(self, url: str, fetch_callback=None, expiry_days: Optional[int] = None) -> Tuple[CachedContent, bool]:
        """
        Get content from cache or fetch new content if not cached or expired
        
        Args:
            url: URL to get content for
            fetch_callback: Async function to fetch content if not cached (should return html_content string)
            expiry_days: Override default expiry days
            
        Returns:
            Tuple of (CachedContent, from_cache: bool)
        """
        await self.initialize()
        
        try:
            # Try to get from cache first
            cached_content = await self.get_cached_content(url)
            
            if cached_content and not cached_content.is_expired():
                # Cache hit - increment access count and return
                await self._increment_access_count(url)
                self._cache_hits += 1
                logger.info(f"Cache HIT for URL: {url} (access count: {cached_content.access_count + 1})")
                return cached_content, True
            
            # Cache miss or expired - fetch new content
            self._cache_misses += 1
            
            if fetch_callback is None:
                logger.warning(f"Cache MISS for URL: {url} but no fetch callback provided")
                if cached_content:
                    # Return expired content if available
                    logger.info(f"Returning expired content for URL: {url}")
                    return cached_content, True
                else:
                    raise ValueError(f"No cached content available for URL: {url} and no fetch callback provided")
            
            # Fetch new content
            logger.info(f"Cache MISS for URL: {url} - fetching new content")
            html_content = await fetch_callback(url)
            
            if not html_content:
                logger.warning(f"Empty content fetched for URL: {url}")
                if cached_content:
                    # Return expired content as fallback
                    return cached_content, True
                raise ValueError(f"No content available for URL: {url}")
            
            # Store new content in cache
            new_cached_content = await self.store_content(url, html_content, expiry_days or self.default_expiry_days)
            logger.info(f"Stored new content in cache for URL: {url}")
            
            return new_cached_content, False
            
        except Exception as e:
            logger.error(f"Error in get_or_fetch_content for URL {url}: {e}")
            raise
    
    async def get_cached_content(self, url: str) -> Optional[CachedContent]:
        """Get cached content for a URL without incrementing access count"""
        await self.initialize()
        
        try:
            doc = await self.mongodb_service.html_cache.find_one({"url": url})
            if doc:
                return CachedContent.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Error getting cached content for URL {url}: {e}")
            return None
    
    async def store_content(self, url: str, html_content: str, expiry_days: int = None) -> CachedContent:
        """Store new HTML content in cache"""
        await self.initialize()
        
        expiry_days = expiry_days or self.default_expiry_days
        cached_content = CachedContent.create_new(url, html_content, expiry_days)
        
        try:
            # Use upsert to handle URL uniqueness
            await self.mongodb_service.html_cache.replace_one(
                {"url": url},
                cached_content.to_dict(),
                upsert=True
            )
            logger.info(f"Stored content for URL: {url} (expires in {expiry_days} days)")
            return cached_content
            
        except Exception as e:
            logger.error(f"Error storing content for URL {url}: {e}")
            raise
    
    async def _increment_access_count(self, url: str) -> int:
        """Increment access count for cached content and update last_accessed"""
        try:
            result = await self.mongodb_service.html_cache.update_one(
                {"url": url},
                {
                    "$inc": {"access_count": 1},
                    "$set": {"last_accessed": datetime.now(timezone.utc)}
                }
            )
            
            if result.modified_count > 0:
                # Get updated access count
                doc = await self.mongodb_service.html_cache.find_one({"url": url}, {"access_count": 1})
                return doc["access_count"] if doc else 1
            else:
                logger.warning(f"Failed to increment access count for URL: {url}")
                return 1
                
        except Exception as e:
            logger.error(f"Error incrementing access count for URL {url}: {e}")
            return 1
    
    async def cleanup_expired_content(self) -> CacheCleanupResponse:
        """Remove expired cache entries and return cleanup statistics"""
        await self.initialize()
        
        try:
            # MongoDB TTL index should handle automatic cleanup, but we can also manually clean
            # and provide statistics
            
            # Count expired entries
            expired_count = await self.mongodb_service.html_cache.count_documents({
                "expiration_date": {"$lt": datetime.now(timezone.utc)}
            })
            
            if expired_count == 0:
                # Get total remaining count
                total_remaining = await self.mongodb_service.html_cache.count_documents({})
                return CacheCleanupResponse(
                    entries_removed=0,
                    space_freed_mb=0.0,
                    total_remaining=total_remaining
                )
            
            # Calculate approximate space to be freed (rough estimate)
            expired_docs = await self.mongodb_service.html_cache.find(
                {"expiration_date": {"$lt": datetime.now(timezone.utc)}},
                {"html_content": 1}
            ).to_list(length=None)
            
            space_freed = sum(len(doc.get("html_content", "").encode('utf-8')) for doc in expired_docs)
            space_freed_mb = space_freed / (1024 * 1024)
            
            # Delete expired entries
            delete_result = await self.mongodb_service.html_cache.delete_many({
                "expiration_date": {"$lt": datetime.now(timezone.utc)}
            })
            
            # Get total remaining count
            total_remaining = await self.mongodb_service.html_cache.count_documents({})
            
            logger.info(f"Cleaned up {delete_result.deleted_count} expired cache entries, freed {space_freed_mb:.2f} MB")
            
            return CacheCleanupResponse(
                entries_removed=delete_result.deleted_count,
                space_freed_mb=space_freed_mb,
                total_remaining=total_remaining
            )
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            raise
    
    async def get_cache_stats(self) -> CacheStatsResponse:
        """Get cache statistics"""
        await self.initialize()
        
        try:
            # Total entries
            total_entries = await self.mongodb_service.html_cache.count_documents({})
            
            # Expired entries
            expired_entries = await self.mongodb_service.html_cache.count_documents({
                "expiration_date": {"$lt": datetime.now(timezone.utc)}
            })
            
            # Calculate cache hit rate
            total_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0
            
            # Calculate total size (approximate)
            all_docs = await self.mongodb_service.html_cache.find({}, {"html_content": 1}).to_list(length=None)
            total_size = sum(len(doc.get("html_content", "").encode('utf-8')) for doc in all_docs)
            total_size_mb = total_size / (1024 * 1024)
            
            # Get oldest and newest entries
            oldest_doc = await self.mongodb_service.html_cache.find_one(
                {}, sort=[("retrieval_timestamp", 1)]
            )
            newest_doc = await self.mongodb_service.html_cache.find_one(
                {}, sort=[("retrieval_timestamp", -1)]
            )
            
            oldest_entry = oldest_doc["retrieval_timestamp"] if oldest_doc else None
            newest_entry = newest_doc["retrieval_timestamp"] if newest_doc else None
            
            return CacheStatsResponse(
                total_entries=total_entries,
                expired_entries=expired_entries,
                cache_hit_rate=cache_hit_rate,
                total_size_mb=total_size_mb,
                oldest_entry=oldest_entry,
                newest_entry=newest_entry
            )
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            raise
    
    async def invalidate_url(self, url: str) -> bool:
        """Remove a specific URL from cache"""
        await self.initialize()
        
        try:
            result = await self.mongodb_service.html_cache.delete_one({"url": url})
            if result.deleted_count > 0:
                logger.info(f"Invalidated cache entry for URL: {url}")
                return True
            else:
                logger.warning(f"No cache entry found to invalidate for URL: {url}")
                return False
                
        except Exception as e:
            logger.error(f"Error invalidating cache for URL {url}: {e}")
            return False
    
    async def update_content(self, url: str, html_content: str, expiry_days: Optional[int] = None) -> CachedContent:
        """Update existing cached content or create new entry"""
        await self.initialize()
        
        # Check if content exists
        existing = await self.get_cached_content(url)
        
        if existing:
            # Preserve access count and increment it
            new_content = CachedContent.create_new(url, html_content, expiry_days or self.default_expiry_days)
            new_content.access_count = existing.access_count + 1
            new_content.last_accessed = datetime.now(timezone.utc)
            
            await self.mongodb_service.html_cache.replace_one(
                {"url": url},
                new_content.to_dict()
            )
            logger.info(f"Updated cached content for URL: {url} (access count: {new_content.access_count})")
            return new_content
        else:
            # Create new entry
            return await self.store_content(url, html_content, expiry_days)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate
        }
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics counters"""
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Performance metrics reset")