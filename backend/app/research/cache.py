#!/usr/bin/env python3
"""
MongoDB Cache Service for Research System
Enhanced caching with smart content management and performance optimization
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from .interfaces import ICacheService, ExtractedContent
from .config import get_config_manager
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# MongoDB libraries
try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError as e:
    logger.error(f"Missing MongoDB libraries: {e}")
    logger.error("Please install: pip install motor pymongo")
    raise


class MongoDBCacheService(ICacheService):
    """MongoDB-based cache service for scraped web content"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.metrics = get_metrics_collector()
        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        
        # Get cache settings
        cache_settings = self.config.get_cache_settings()
        self.database_name = cache_settings['database_name']
        self.collection_name = cache_settings['collection_name']
        self.cache_expiry_days = self.config.get_research_settings()['cache_expiry_days']
        self.enabled = cache_settings['enabled']
        
        if not self.enabled:
            logger.info("MongoDB cache is disabled in configuration")
    
    async def connect(self):
        """Connect to MongoDB"""
        if not self.enabled:
            logger.info("Cache is disabled, skipping MongoDB connection")
            return
        
        timing_id = self.metrics.start_timing("mongodb_connect")
        
        try:
            mongodb_creds = self.config.get_api_credentials('mongodb')
            mongodb_uri = mongodb_creds['uri']
            
            self.client = AsyncIOMotorClient(mongodb_uri)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Test connection
            await self.client.admin.command('ping')
            self.is_connected = True
            
            # Create indexes for better performance
            await self._create_indexes()
            
            logger.info("✅ Connected to MongoDB cache")
            self.metrics.increment_counter("mongodb_connections_successful")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.is_connected = False
            self.metrics.increment_counter("mongodb_connections_failed")
            raise
        finally:
            self.metrics.end_timing(timing_id)
    
    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            cache_settings = self.config.get_cache_settings()
            index_fields = cache_settings['index_fields']
            
            # Create individual indexes
            for field in index_fields:
                if field == "url":
                    await self.collection.create_index("url", unique=True)
                else:
                    await self.collection.create_index(field)
            
            # Create compound indexes for common queries
            await self.collection.create_index([
                ("keywords", 1),
                ("accessed_date", -1)
            ])
            
            await self.collection.create_index([
                ("domain_info.quality_score", -1),
                ("accessed_date", -1)
            ])
            
            logger.info(f"Created indexes for fields: {index_fields}")
            
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    async def get_cached_content(self, url: str) -> Optional[ExtractedContent]:
        """Get cached content by URL"""
        if not self.enabled or not self.is_connected:
            return None
        
        timing_id = self.metrics.start_timing("cache_get", {"operation": "get_by_url"})
        
        try:
            result = await self.collection.find_one({"url": url})
            
            if result:
                # Check freshness
                accessed_date = result.get('accessed_date')
                if isinstance(accessed_date, datetime):
                    days_old = (datetime.utcnow() - accessed_date).days
                    if days_old <= self.cache_expiry_days:
                        logger.info(f"💾 Cache hit: {url} ({days_old} days old)")
                        self.metrics.increment_counter("cache_hits")
                        
                        # Convert to ExtractedContent
                        content = self._document_to_extracted_content(result)
                        return content
                    else:
                        logger.info(f"⏰ Cache expired: {url} ({days_old} days old)")
                        # Remove expired cache
                        await self.collection.delete_one({"url": url})
                        self.metrics.increment_counter("cache_expired")
            
            self.metrics.increment_counter("cache_misses")
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache retrieval failed for {url}: {e}")
            self.metrics.increment_counter("cache_errors")
            return None
        finally:
            self.metrics.end_timing(timing_id)
    
    async def save_content(self, content: ExtractedContent, keywords: List[str]) -> bool:
        """Save content to cache"""
        if not self.enabled or not self.is_connected:
            return False
        
        timing_id = self.metrics.start_timing("cache_save", {"operation": "save_content"})
        
        try:
            # Parse domain info
            domain = urlparse(content.url).netloc.lower()
            
            document = {
                "url": content.url,
                "title": content.title,
                "content": content.content,
                "keywords": keywords,
                "accessed_date": datetime.utcnow(),
                "method": content.method,
                "word_count": content.word_count,
                "domain_info": content.domain_info or {},
                "domain": domain,
                "extraction_time": content.extraction_time,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Use upsert to handle duplicates
            await self.collection.replace_one(
                {"url": content.url}, 
                document, 
                upsert=True
            )
            
            logger.info(f"💾 Cached content: {content.url} ({content.word_count} words)")
            self.metrics.increment_counter("cache_saves")
            self.metrics.record_histogram_value("cache_content_size", content.word_count)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache save failed for {content.url}: {e}")
            self.metrics.increment_counter("cache_save_errors")
            return False
        finally:
            self.metrics.end_timing(timing_id)
    
    async def search_cached_content(self, keywords: List[str]) -> List[ExtractedContent]:
        """Search for cached content by keywords"""
        if not self.enabled or not self.is_connected:
            return []
        
        timing_id = self.metrics.start_timing("cache_search", {"keywords_count": len(keywords)})
        
        try:
            # Build search query for keywords
            query = {
                "$and": [
                    {
                        "$or": [
                            {"keywords": {"$in": keywords}},
                            {"title": {"$regex": "|".join(keywords), "$options": "i"}},
                            {"content": {"$regex": "|".join(keywords), "$options": "i"}}
                        ]
                    },
                    {
                        "accessed_date": {
                            "$gte": datetime.utcnow() - timedelta(days=self.cache_expiry_days)
                        }
                    }
                ]
            }
            
            # Sort by quality score and recency
            cursor = self.collection.find(query).sort([
                ("domain_info.quality_score", -1),
                ("accessed_date", -1)
            ]).limit(20)  # Limit results to prevent overwhelming
            
            cached_results = []
            async for doc in cursor:
                content = self._document_to_extracted_content(doc)
                cached_results.append(content)
                
                days_old = (datetime.utcnow() - doc.get('accessed_date', datetime.utcnow())).days
                logger.info(f"🔍 Found cached content: {doc['url']} ({days_old} days old)")
            
            logger.info(f"📊 Found {len(cached_results)} cached results for keywords: {keywords}")
            self.metrics.increment_counter("cache_searches")
            self.metrics.record_histogram_value("cache_search_results", len(cached_results))
            
            return cached_results
            
        except Exception as e:
            logger.error(f"❌ Cache search failed: {e}")
            self.metrics.increment_counter("cache_search_errors")
            return []
        finally:
            self.metrics.end_timing(timing_id)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled or not self.is_connected:
            return {"total_entries": 0, "connected": False, "enabled": False}
        
        timing_id = self.metrics.start_timing("cache_stats")
        
        try:
            # Total entries
            total_entries = await self.collection.count_documents({})
            
            # Fresh entries (within expiry period)
            fresh_entries = await self.collection.count_documents({
                "accessed_date": {"$gte": datetime.utcnow() - timedelta(days=self.cache_expiry_days)}
            })
            
            # Entries by domain type
            domain_pipeline = [
                {"$group": {"_id": "$domain_info.source_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            domain_stats = []
            async for doc in self.collection.aggregate(domain_pipeline):
                domain_stats.append({"type": doc["_id"], "count": doc["count"]})
            
            # Average content size
            size_pipeline = [
                {"$group": {"_id": None, "avg_size": {"$avg": "$word_count"}}}
            ]
            avg_size = 0
            async for doc in self.collection.aggregate(size_pipeline):
                avg_size = doc.get("avg_size", 0)
            
            # Recent activity (last 24 hours)
            recent_activity = await self.collection.count_documents({
                "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
            })
            
            stats = {
                "total_entries": total_entries,
                "fresh_entries": fresh_entries,
                "expired_entries": total_entries - fresh_entries,
                "domain_distribution": domain_stats,
                "average_content_size": round(avg_size, 2),
                "recent_activity_24h": recent_activity,
                "cache_expiry_days": self.cache_expiry_days,
                "connected": True,
                "enabled": True
            }
            
            self.metrics.set_gauge("cache_total_entries", total_entries)
            self.metrics.set_gauge("cache_fresh_entries", fresh_entries)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Cache stats failed: {e}")
            return {"total_entries": 0, "connected": False, "error": str(e)}
        finally:
            self.metrics.end_timing(timing_id)
    
    async def cleanup_expired_content(self) -> int:
        """Clean up expired content and return number of deleted entries"""
        if not self.enabled or not self.is_connected:
            return 0
        
        timing_id = self.metrics.start_timing("cache_cleanup")
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.cache_expiry_days)
            
            result = await self.collection.delete_many({
                "accessed_date": {"$lt": cutoff_date}
            })
            
            deleted_count = result.deleted_count
            logger.info(f"🧹 Cleaned up {deleted_count} expired cache entries")
            self.metrics.increment_counter("cache_cleanup_entries", deleted_count)
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")
            return 0
        finally:
            self.metrics.end_timing(timing_id)
    
    async def get_content_by_domain(self, domain: str, limit: int = 10) -> List[ExtractedContent]:
        """Get cached content for a specific domain"""
        if not self.enabled or not self.is_connected:
            return []
        
        try:
            cursor = self.collection.find({
                "domain": domain,
                "accessed_date": {"$gte": datetime.utcnow() - timedelta(days=self.cache_expiry_days)}
            }).sort("accessed_date", -1).limit(limit)
            
            results = []
            async for doc in cursor:
                content = self._document_to_extracted_content(doc)
                results.append(content)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get content for domain {domain}: {e}")
            return []
    
    def _document_to_extracted_content(self, doc: Dict[str, Any]) -> ExtractedContent:
        """Convert MongoDB document to ExtractedContent"""
        return ExtractedContent(
            url=doc['url'],
            title=doc.get('title', 'No title'),
            content=doc.get('content', ''),
            method=doc.get('method', 'cache'),
            word_count=doc.get('word_count', 0),
            success=True,
            extraction_time=doc.get('extraction_time', 0.0),
            domain_info=doc.get('domain_info', {}),
            from_cache=True,
            cache_date=doc.get('accessed_date')
        )
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("🔌 MongoDB connection closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Factory function for creating cache service
def create_cache_service() -> ICacheService:
    """Create and return a cache service instance"""
    config = get_config_manager()
    cache_settings = config.get_cache_settings()
    
    if cache_settings['enabled']:
        return MongoDBCacheService()
    else:
        # Return a no-op cache service if caching is disabled
        return NoOpCacheService()


class NoOpCacheService(ICacheService):
    """No-operation cache service for when caching is disabled"""
    
    async def get_cached_content(self, url: str) -> Optional[ExtractedContent]:
        return None
    
    async def save_content(self, content: ExtractedContent, keywords: List[str]) -> bool:
        return True
    
    async def search_cached_content(self, keywords: List[str]) -> List[ExtractedContent]:
        return []
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        return {"total_entries": 0, "connected": False, "enabled": False}
