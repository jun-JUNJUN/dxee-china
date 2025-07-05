#!/usr/bin/env python3
"""
Enhanced Content Extractor Service
Advanced web content extraction with quality assessment and caching integration
"""

import logging
import requests
import time
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from .interfaces import IContentExtractor, ExtractedContent, ICacheService
from .config import get_config_manager
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# Web scraping libraries
try:
    from bs4 import BeautifulSoup
    import newspaper
    from newspaper import Article
    from readability import Document
except ImportError as e:
    logger.error(f"Missing required libraries: {e}")
    logger.error("Please install: pip install beautifulsoup4 lxml newspaper3k readability-lxml")
    raise


class EnhancedWebContentExtractor(IContentExtractor):
    """Enhanced service for extracting content from web pages with caching and quality assessment"""
    
    def __init__(self, cache_service: ICacheService):
        self.config = get_config_manager()
        self.metrics = get_metrics_collector()
        self.cache_service = cache_service
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Domain quality cache to avoid repeated assessments
        self.domain_quality_cache = {}
        
        # Get extraction settings
        research_settings = self.config.get_research_settings()
        self.extraction_timeout = research_settings['extraction_timeout']
        self.max_content_length = research_settings['max_content_length']
        self.request_delay = research_settings['request_delay_seconds']
    
    async def extract_content(self, url: str, keywords: List[str] = None) -> ExtractedContent:
        """Extract content from a URL with caching support"""
        timing_id = self.metrics.start_timing("content_extraction", {
            "domain": urlparse(url).netloc
        })
        
        try:
            logger.info(f"🔄 Extracting content from: {url}")
            
            # Check cache first
            cached_content = await self.cache_service.get_cached_content(url)
            if cached_content:
                extraction_time = time.time() - time.time()  # Minimal time for cache hit
                logger.info(f"💾 Using cached content: {len(cached_content.content)} chars")
                self.metrics.increment_counter("content_cache_hits")
                
                # Update timing and return
                cached_content.extraction_time = extraction_time
                return cached_content
            
            # Assess domain quality first
            domain_info = self.assess_domain_quality(url)
            
            # Extract content using various methods
            extraction_result = await self._extract_content_methods(url, domain_info)
            
            # Save to cache if extraction was successful and keywords provided
            if extraction_result.success and keywords and self.cache_service:
                await self.cache_service.save_content(extraction_result, keywords)
            
            # Record metrics
            self.metrics.increment_counter("content_extractions")
            if extraction_result.success:
                self.metrics.increment_counter("content_extractions_successful")
                self.metrics.record_histogram_value("content_word_count", extraction_result.word_count)
            else:
                self.metrics.increment_counter("content_extractions_failed")
            
            return extraction_result
            
        except Exception as e:
            extraction_time = time.time() - time.time()
            logger.error(f"❌ Content extraction failed for {url}: {e}")
            self.metrics.increment_counter("content_extraction_errors")
            
            return ExtractedContent(
                url=url,
                title='Error',
                content=f'Error extracting content: {str(e)}',
                method='error',
                word_count=0,
                success=False,
                extraction_time=extraction_time,
                domain_info={'quality_score': 0, 'source_type': 'error'},
                error=str(e),
                from_cache=False
            )
        finally:
            self.metrics.end_timing(timing_id)
    
    def assess_domain_quality(self, url: str) -> Dict[str, Any]:
        """Assess the quality and type of a domain for content extraction"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check cache first
            if domain in self.domain_quality_cache:
                return self.domain_quality_cache[domain]
            
            # High-quality business/news domains
            high_quality_indicators = [
                'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'economist.com',
                'forbes.com', 'fortune.com', 'businessweek.com', 'harvard.edu',
                'mckinsey.com', 'bcg.com', 'deloitte.com', 'pwc.com', 'kpmg.com',
                'gartner.com', 'forrester.com', 'idc.com', 'statista.com',
                'techcrunch.com', 'venturebeat.com', 'wired.com', 'arstechnica.com'
            ]
            
            # Medium-quality sources
            medium_quality_indicators = [
                'zdnet.com', 'computerworld.com', 'infoworld.com', 'cio.com',
                'wikipedia.org', 'investopedia.com', 'businessinsider.com',
                'cnbc.com', 'marketwatch.com', 'fool.com'
            ]
            
            # Low-quality or problematic sources
            low_quality_indicators = [
                'reddit.com', 'quora.com', 'yahoo.com/answers', 'stackoverflow.com',
                'medium.com', 'linkedin.com/pulse', 'facebook.com', 'twitter.com',
                'instagram.com', 'tiktok.com', 'pinterest.com'
            ]
            
            quality_score = 5  # Default medium quality
            source_type = "general"
            extraction_priority = "medium"
            
            # Assess quality
            if any(indicator in domain for indicator in high_quality_indicators):
                quality_score = 9
                source_type = "premium"
                extraction_priority = "high"
            elif any(indicator in domain for indicator in medium_quality_indicators):
                quality_score = 7
                source_type = "reliable"
                extraction_priority = "medium"
            elif any(indicator in domain for indicator in low_quality_indicators):
                quality_score = 3
                source_type = "social"
                extraction_priority = "low"
            elif domain.endswith('.edu') or domain.endswith('.gov'):
                quality_score = 8
                source_type = "academic"
                extraction_priority = "high"
            elif domain.endswith('.org'):
                quality_score = 6
                source_type = "organization"
                extraction_priority = "medium"
            
            # Company websites (ending in .com with short domains)
            if domain.endswith('.com') and len(domain.split('.')) == 2 and len(domain.split('.')[0]) < 15:
                quality_score = max(quality_score, 6)
                source_type = "corporate"
                extraction_priority = "medium"
            
            # Additional quality indicators based on URL structure
            if '/press-release/' in url or '/news/' in url or '/report/' in url:
                quality_score += 1
                extraction_priority = "high"
            
            if '/blog/' in url or '/opinion/' in url or '/personal/' in url:
                quality_score -= 1
                extraction_priority = "low"
            
            result = {
                'domain': domain,
                'quality_score': min(10, max(1, quality_score)),
                'source_type': source_type,
                'extraction_priority': extraction_priority,
                'recommended': quality_score >= 6
            }
            
            self.domain_quality_cache[domain] = result
            logger.info(f"🔍 Domain assessment: {domain} - Quality: {quality_score}/10, Type: {source_type}")
            
            # Record metrics
            self.metrics.record_histogram_value("domain_quality_score", quality_score)
            self.metrics.increment_counter("domain_assessments", tags={"source_type": source_type})
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Domain assessment failed for {url}: {e}")
            return {
                'domain': 'unknown', 
                'quality_score': 5, 
                'source_type': 'unknown', 
                'extraction_priority': 'medium',
                'recommended': True
            }
    
    async def _extract_content_methods(self, url: str, domain_info: Dict[str, Any]) -> ExtractedContent:
        """Try different content extraction methods based on domain quality"""
        start_time = time.time()
        
        # Choose extraction strategy based on domain quality
        extraction_priority = domain_info.get('extraction_priority', 'medium')
        
        if extraction_priority == 'high':
            # For high-quality sources, try all methods
            methods = ['newspaper3k', 'readability', 'beautifulsoup']
        elif extraction_priority == 'low':
            # For low-quality sources, use simpler methods
            methods = ['beautifulsoup', 'readability']
        else:
            # Default order
            methods = ['newspaper3k', 'readability', 'beautifulsoup']
        
        for method in methods:
            try:
                if method == 'newspaper3k':
                    result = await self._extract_with_newspaper(url, domain_info, start_time)
                elif method == 'readability':
                    result = await self._extract_with_readability(url, domain_info, start_time)
                elif method == 'beautifulsoup':
                    result = await self._extract_with_beautifulsoup(url, domain_info, start_time)
                else:
                    continue
                
                if result and result.success:
                    self.metrics.increment_counter("extraction_method_success", tags={"method": method})
                    return result
                    
            except Exception as e:
                logger.warning(f"⚠️ {method} extraction failed for {url}: {e}")
                self.metrics.increment_counter("extraction_method_failed", tags={"method": method})
                continue
        
        # All methods failed
        extraction_time = time.time() - start_time
        logger.error(f"❌ All extraction methods failed for {url} in {extraction_time:.2f}s")
        
        return ExtractedContent(
            url=url,
            title='Extraction failed',
            content='Could not extract content from this URL',
            method='none',
            word_count=0,
            success=False,
            extraction_time=extraction_time,
            domain_info=domain_info,
            error='All extraction methods failed',
            from_cache=False
        )
    
    async def _extract_with_newspaper(self, url: str, domain_info: Dict[str, Any], start_time: float) -> Optional[ExtractedContent]:
        """Extract content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text.strip()) > 100:
                extraction_time = time.time() - start_time
                content = article.text[:self.max_content_length] if len(article.text) > self.max_content_length else article.text
                
                logger.info(f"✅ Successful extraction via newspaper3k: {len(content)} chars in {extraction_time:.2f}s")
                
                return ExtractedContent(
                    url=url,
                    title=article.title or 'No title',
                    content=content,
                    method='newspaper3k',
                    word_count=len(content.split()),
                    success=True,
                    extraction_time=extraction_time,
                    domain_info=domain_info,
                    from_cache=False
                )
        except Exception as e:
            logger.warning(f"⚠️ Newspaper3k failed for {url}: {e}")
            return None
    
    async def _extract_with_readability(self, url: str, domain_info: Dict[str, Any], start_time: float) -> Optional[ExtractedContent]:
        """Extract content using readability + BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=self.extraction_timeout)
            response.raise_for_status()
            
            doc = Document(response.text)
            soup = BeautifulSoup(doc.content(), 'html.parser')
            
            content = soup.get_text(separator=' ', strip=True)
            title = doc.title() or soup.find('title')
            title = title.get_text() if hasattr(title, 'get_text') else str(title) if title else 'No title'
            
            if content and len(content.strip()) > 100:
                extraction_time = time.time() - start_time
                content = content[:self.max_content_length] if len(content) > self.max_content_length else content
                
                logger.info(f"✅ Successful extraction via readability: {len(content)} chars in {extraction_time:.2f}s")
                
                return ExtractedContent(
                    url=url,
                    title=title,
                    content=content,
                    method='readability+bs4',
                    word_count=len(content.split()),
                    success=True,
                    extraction_time=extraction_time,
                    domain_info=domain_info,
                    from_cache=False
                )
        except Exception as e:
            logger.warning(f"⚠️ Readability method failed for {url}: {e}")
            return None
    
    async def _extract_with_beautifulsoup(self, url: str, domain_info: Dict[str, Any], start_time: float) -> Optional[ExtractedContent]:
        """Extract content using BeautifulSoup fallback"""
        try:
            response = self.session.get(url, timeout=self.extraction_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'article', 'main', '.content', '.article-content', 
                '.post-content', '.entry-content', '#content',
                '.story-body', '.article-body', '.post-body'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    break
            
            # If no specific content area found, get body text
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            title = soup.find('title')
            title = title.get_text() if title else 'No title'
            
            if content and len(content.strip()) > 50:
                extraction_time = time.time() - start_time
                content = content[:self.max_content_length] if len(content) > self.max_content_length else content
                
                logger.info(f"✅ Successful extraction via BeautifulSoup: {len(content)} chars in {extraction_time:.2f}s")
                
                return ExtractedContent(
                    url=url,
                    title=title,
                    content=content,
                    method='beautifulsoup',
                    word_count=len(content.split()),
                    success=True,
                    extraction_time=extraction_time,
                    domain_info=domain_info,
                    from_cache=False
                )
        except Exception as e:
            logger.warning(f"⚠️ BeautifulSoup method failed for {url}: {e}")
            return None
    
    async def extract_multiple_contents(self, urls: List[str], keywords: List[str] = None) -> List[ExtractedContent]:
        """Extract content from multiple URLs with intelligent batching"""
        timing_id = self.metrics.start_timing("multi_content_extraction", {
            "url_count": str(len(urls))
        })
        
        logger.info(f"📄 Extracting content from {len(urls)} sources")
        
        # Sort URLs by domain quality for better extraction order
        url_quality_pairs = []
        for url in urls:
            domain_info = self.assess_domain_quality(url)
            url_quality_pairs.append((url, domain_info['quality_score']))
        
        # Sort by quality score (highest first)
        sorted_urls = [url for url, _ in sorted(url_quality_pairs, key=lambda x: x[1], reverse=True)]
        
        extracted_contents = []
        
        for i, url in enumerate(sorted_urls):
            logger.info(f"📄 Extracting {i+1}/{len(sorted_urls)}: {urlparse(url).netloc}")
            
            content = await self.extract_content(url, keywords)
            extracted_contents.append(content)
            
            if content.success:
                cache_status = "💾 Cache" if content.from_cache else "🌐 Live"
                quality_score = content.domain_info.get('quality_score', 0)
                logger.info(f"  ✅ Success ({cache_status}): {content.word_count} words, quality: {quality_score}/10")
            else:
                logger.warning(f"  ❌ Failed: {content.error}")
            
            # Add delay only for live scraping (not cached content)
            if not content.from_cache and i < len(sorted_urls) - 1:
                await asyncio.sleep(self.request_delay)
        
        successful_extractions = sum(1 for c in extracted_contents if c.success)
        logger.info(f"✅ Content extraction completed: {successful_extractions}/{len(extracted_contents)} successful")
        
        self.metrics.record_histogram_value("multi_extraction_success_rate", 
                                          (successful_extractions / len(extracted_contents)) * 100)
        self.metrics.end_timing(timing_id)
        
        return extracted_contents
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get content extraction statistics"""
        return {
            'domain_cache_size': len(self.domain_quality_cache),
            'extraction_timeout': self.extraction_timeout,
            'max_content_length': self.max_content_length,
            'request_delay': self.request_delay
        }
    
    def clear_domain_cache(self):
        """Clear the domain quality cache"""
        self.domain_quality_cache.clear()
        logger.info("Domain quality cache cleared")


# Factory function for creating content extractor
def create_content_extractor(cache_service: ICacheService) -> IContentExtractor:
    """Create and return a content extractor instance"""
    return EnhancedWebContentExtractor(cache_service)
