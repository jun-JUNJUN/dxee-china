#!/usr/bin/env python3
"""
Cache Models - Data models for HTML content caching with access counters
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import hashlib


@dataclass
class CachedContent:
    """Dataclass for cached HTML content with access tracking"""
    url: str
    html_content: str
    retrieval_timestamp: datetime
    content_hash: str
    expiration_date: datetime
    access_count: int
    last_accessed: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_new(cls, url: str, html_content: str, expiry_days: int = 30) -> 'CachedContent':
        """Create a new cached content entry"""
        now = datetime.now(timezone.utc)
        content_hash = hashlib.sha256(html_content.encode('utf-8')).hexdigest()
        expiration_date = datetime.now(timezone.utc).replace(microsecond=0) + \
                         timedelta(days=expiry_days)
        
        return cls(
            url=url,
            html_content=html_content,
            retrieval_timestamp=now,
            content_hash=content_hash,
            expiration_date=expiration_date,
            access_count=1,  # First access
            last_accessed=now,
            metadata={}
        )
    
    def increment_access(self) -> None:
        """Increment access count and update last accessed timestamp"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if the cached content has expired"""
        return datetime.now(timezone.utc) > self.expiration_date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'url': self.url,
            'html_content': self.html_content,
            'retrieval_timestamp': self.retrieval_timestamp,
            'content_hash': self.content_hash,
            'expiration_date': self.expiration_date,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedContent':
        """Create from dictionary (MongoDB document)"""
        return cls(
            url=data['url'],
            html_content=data['html_content'],
            retrieval_timestamp=data['retrieval_timestamp'],
            content_hash=data['content_hash'],
            expiration_date=data['expiration_date'],
            access_count=data['access_count'],
            last_accessed=data['last_accessed'],
            metadata=data.get('metadata', {})
        )


class CachedContentRequest(BaseModel):
    """Pydantic model for cache requests"""
    url: str = Field(..., description="URL to cache content for", min_length=1)
    html_content: str = Field(..., description="HTML content to cache")
    expiry_days: int = Field(default=30, description="Cache expiry in days", ge=1, le=365)


class CachedContentResponse(BaseModel):
    """Pydantic model for cache responses"""
    url: str = Field(..., description="URL of cached content")
    html_content: str = Field(..., description="Cached HTML content")
    retrieval_timestamp: datetime = Field(..., description="When content was originally retrieved")
    content_hash: str = Field(..., description="SHA256 hash of content")
    expiration_date: datetime = Field(..., description="When cache expires")
    access_count: int = Field(..., description="Number of times accessed")
    last_accessed: datetime = Field(..., description="Last access timestamp")
    is_expired: bool = Field(..., description="Whether content has expired")
    from_cache: bool = Field(default=True, description="Whether this came from cache")


class CacheStatsResponse(BaseModel):
    """Pydantic model for cache statistics"""
    total_entries: int = Field(..., description="Total cache entries")
    expired_entries: int = Field(..., description="Number of expired entries")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    total_size_mb: float = Field(..., description="Total cache size in MB")
    oldest_entry: Optional[datetime] = Field(None, description="Oldest cache entry timestamp")
    newest_entry: Optional[datetime] = Field(None, description="Newest cache entry timestamp")


class CacheCleanupResponse(BaseModel):
    """Pydantic model for cache cleanup results"""
    entries_removed: int = Field(..., description="Number of expired entries removed")
    space_freed_mb: float = Field(..., description="Space freed in MB")
    total_remaining: int = Field(..., description="Total entries remaining")