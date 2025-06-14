#!/usr/bin/env python3
"""
Dual Research Service - Runs both intelligent and legacy research approaches
Based on test_deepseek_advanced_web_research.py algorithm
"""

import os
import json
import asyncio
import logging
import requests
import re
from datetime import datetime
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urljoin, urlparse
import time

# Web scraping libraries
try:
    from bs4 import BeautifulSoup
    import newspaper
    from newspaper import Article
    from readability import Document
except ImportError as e:
    logging.error(f"Missing required libraries: {e}")
    logging.error("Please install: pip install beautifulsoup4 lxml newspaper3k readability-lxml")
    raise

logger = logging.getLogger(__name__)

class WebContentExtractor:
    """Service for extracting content from web pages"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def extract_article_content(self, url: str) -> Dict[str, Any]:
        """
        Extract article content from a URL using multiple methods
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with extracted content, title, and metadata
        """
        try:
            logger.info(f"Extracting content from: {url}")
            
            # Method 1: Try newspaper3k first (best for articles)
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                if article.text and len(article.text.strip()) > 100:
                    return {
                        'url': url,
                        'title': article.title or 'No title',
                        'content': article.text,
                        'method': 'newspaper3k',
                        'word_count': len(article.text.split()),
                        'success': True
                    }
            except Exception as e:
                logger.warning(f"Newspaper3k failed for {url}: {e}")
            
            # Method 2: Try readability + BeautifulSoup
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                # Use readability to extract main content
                doc = Document(response.text)
                soup = BeautifulSoup(doc.content(), 'html.parser')
                
                # Extract text content
                content = soup.get_text(separator=' ', strip=True)
                title = doc.title() or soup.find('title')
                title = title.get_text() if hasattr(title, 'get_text') else str(title) if title else 'No title'
                
                if content and len(content.strip()) > 100:
                    return {
                        'url': url,
                        'title': title,
                        'content': content,
                        'method': 'readability+bs4',
                        'word_count': len(content.split()),
                        'success': True
                    }
            except Exception as e:
                logger.warning(f"Readability method failed for {url}: {e}")
            
            # Method 3: Basic BeautifulSoup fallback
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Try to find main content areas
                content_selectors = [
                    'article', 'main', '.content', '.article-content', 
                    '.post-content', '.entry-content', '#content'
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
                    return {
                        'url': url,
                        'title': title,
                        'content': content[:5000],  # Limit content length
                        'method': 'beautifulsoup',
                        'word_count': len(content.split()),
                        'success': True
                    }
            except Exception as e:
                logger.warning(f"BeautifulSoup method failed for {url}: {e}")
            
            # All methods failed
            return {
                'url': url,
                'title': 'Extraction failed',
                'content': 'Could not extract content from this URL',
                'method': 'none',
                'word_count': 0,
                'success': False,
                'error': 'All extraction methods failed'
            }
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return {
                'url': url,
                'title': 'Error',
                'content': f'Error extracting content: {str(e)}',
                'method': 'error',
                'word_count': 0,
                'success': False,
                'error': str(e)
            }

class GoogleWebSearchService:
    """Enhanced Google web search service"""
    
    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. Web search functionality will be limited.")
        if not self.cse_id:
            logger.warning("GOOGLE_CSE_ID not set. Web search functionality will be limited.")
    
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a Google web search
        
        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            
        Returns:
            List of search results with title, link, snippet
        """
        if not self.api_key or not self.cse_id:
            logger.error("Google API credentials not configured")
            return []
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
            logger.info(f"Performing Google search for: {query}")
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'displayLink': item.get('displayLink', '')
                    })
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []

class DualResearchService:
    """
    Dual research service that runs both intelligent and legacy research approaches
    Based on the algorithm from test_deepseek_advanced_web_research.py
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.web_search = GoogleWebSearchService()
        self.content_extractor = WebContentExtractor()
        self.progress_callback = progress_callback
        
        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    def _notify_progress(self, step: str, data: Dict[str, Any]):
        """Notify progress callback if available"""
        if self.progress_callback:
            self.progress_callback(step, data)

    async def step0_check_web_search_necessity(self, original_question: str) -> Dict[str, Any]:
        """Step 0: Check if web search is necessary for answering the question"""
        try:
            logger.info("Step 0: Checking if web search is necessary")
            
            system_message = """You are an expert research assistant. Your task is to analyze a question and determine whether web search is necessary to provide a comprehensive and accurate answer.

Instructions:
1. Analyze the given question carefully
2. Determine if the question requires current/recent information, specific data, or real-time facts that would benefit from web search
3. If web search IS needed, respond with: Query="your_optimized_search_query_here"
4. If web search is NOT needed, provide a comprehensive answer directly

Questions that typically NEED web search:
- Current events, recent news, latest developments
- Specific company information, financial data, market rankings
- Real-time statistics, current prices, recent reports
- Location-specific information that changes frequently
- Technical specifications of recent products

Questions that typically DON'T need web search:
- General knowledge questions
- Historical facts (unless very recent)
- Mathematical calculations
- Theoretical concepts and explanations
- Programming concepts and syntax
- Well-established scientific principles"""

            user_prompt = f"""Question: {original_question}

Please analyze this question and determine if web search is necessary. If web search is needed, respond with Query="search_terms". If not needed, provide a comprehensive answer directly."""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"DeepSeek web search necessity check: {response_text[:200]}...")
            
            # Check if response contains a search query
            query_match = re.search(r'Query="([^"]+)"', response_text)
            if query_match:
                search_query = query_match.group(1)
                logger.info(f"Web search needed. Extracted search query: {search_query}")
                return {
                    'web_search_needed': True,
                    'search_query': search_query,
                    'response': response_text
                }
            else:
                logger.info("Web search not needed. Direct answer provided.")
                return {
                    'web_search_needed': False,
                    'direct_answer': response_text,
                    'response': response_text
                }
                
        except Exception as e:
            logger.error(f"Step 0 failed: {e}")
            # Default to web search if check fails
            return {
                'web_search_needed': True,
                'search_query': original_question,
                'error': str(e)
            }

    async def step1_generate_search_query(self, original_question: str) -> str:
        """Step 1: Use DeepSeek to generate optimal search query (legacy method)"""
        try:
            logger.info("Step 1: Generating optimal search query")
            
            system_message = """You are an expert research assistant. Your task is to analyze a research question and generate the most effective search query for finding relevant information on the web.

Instructions:
1. Analyze the given question carefully
2. Identify the key concepts and search terms
3. Create an optimized search query that will find the most relevant results
4. Return your response in the exact format: Query="your_search_query_here"
5. Make the search query specific enough to find relevant results but broad enough to capture comprehensive information
6. Consider using industry-specific terms, location modifiers, and relevant keywords"""

            user_prompt = f"""Original Question: {original_question}

Please generate the optimal search query for finding comprehensive information to answer this question. Remember to respond in the format Query="your_search_query_here"."""

            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"DeepSeek query generation response: {response_text}")
            
            # Extract query from response
            query_match = re.search(r'Query="([^"]+)"', response_text)
            if query_match:
                search_query = query_match.group(1)
                logger.info(f"Extracted search query: {search_query}")
                return search_query
            else:
                # Fallback: use the original question
                logger.warning("Could not extract query from response, using original question")
                return original_question
                
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            return original_question

    async def step2_web_search(self, search_query: str) -> List[Dict[str, Any]]:
        """Step 2: Perform Google web search"""
        logger.info("Step 2: Performing web search")
        return await self.web_search.search(search_query, num_results=5)

    async def step3_extract_content(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Extract full content from search result pages"""
        logger.info("Step 3: Extracting content from search results")
        
        extracted_contents = []
        for i, result in enumerate(search_results):
            logger.info(f"Extracting content from result {i+1}/{len(search_results)}: {result['link']}")
            
            content = await self.content_extractor.extract_article_content(result['link'])
            content['search_result'] = result  # Include original search result
            extracted_contents.append(content)
            
            # Add delay to be respectful to websites
            await asyncio.sleep(1)
        
        return extracted_contents

    async def step4_analyze_relevance(self, original_question: str, extracted_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 4: Use DeepSeek Reasoning to analyze content relevance"""
        logger.info("Step 4: Analyzing content relevance with DeepSeek Reasoning")
        
        try:
            # Prepare content for analysis
            content_summaries = []
            for i, content in enumerate(extracted_contents):
                if content['success']:
                    summary = f"""
Source {i+1}: {content['title']}
URL: {content['url']}
Content Preview: {content['content'][:1000]}...
Word Count: {content['word_count']}
"""
                else:
                    summary = f"""
Source {i+1}: {content['title']} (Extraction Failed)
URL: {content['url']}
Error: {content.get('error', 'Unknown error')}
"""
                content_summaries.append(summary)
            
            system_message = """You are an expert research analyst with advanced reasoning capabilities. Your task is to analyze extracted web content and determine its relevance to answering a specific research question.

Instructions:
1. Carefully analyze each piece of extracted content
2. Determine how well each source answers the original question
3. Identify which sources provide the most relevant and accurate information
4. For each source, provide:
   - Relevance score (1-10, where 10 is highly relevant)
   - Key insights that help answer the question
   - Any limitations or concerns about the source
5. Synthesize the information to provide a comprehensive answer
6. If there are discrepancies between sources, analyze the causes
7. Provide your reasoning process clearly"""

            user_prompt = f"""Original Research Question: {original_question}

Extracted Content from Web Sources:
{''.join(content_summaries)}

Please analyze each source for relevance to the original question and provide:
1. Individual source analysis with relevance scores
2. Key findings that answer the original question
3. Synthesis of information across sources
4. Analysis of any discrepancies or conflicting information
5. Overall assessment of how well the sources answer the original question"""

            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=60.0
            )
            
            # Extract reasoning and answer content
            message = response.choices[0].message
            reasoning_content = getattr(message, 'reasoning_content', '') if hasattr(message, 'reasoning_content') else ''
            analysis_content = message.content or ''
            
            return {
                'original_question': original_question,
                'analysis_content': analysis_content,
                'reasoning_content': reasoning_content,
                'sources_analyzed': len(extracted_contents),
                'successful_extractions': sum(1 for c in extracted_contents if c['success']),
                'model': response.model,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Step 4 analysis failed: {e}")
            return {
                'original_question': original_question,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def conduct_intelligent_research(self, original_question: str) -> Dict[str, Any]:
        """Conduct intelligent research process with web search necessity check"""
        logger.info(f"Starting intelligent research for: {original_question}")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'research_type': 'intelligent',
            'steps': {}
        }
        
        try:
            # Step 0: Check if web search is necessary
            necessity_check = await self.step0_check_web_search_necessity(original_question)
            results['steps']['step0'] = {
                'description': 'Check web search necessity',
                'web_search_needed': necessity_check['web_search_needed'],
                'response': necessity_check['response'],
                'success': True
            }
            
            # If web search is not needed, return direct answer
            if not necessity_check['web_search_needed']:
                results['direct_answer'] = necessity_check['direct_answer']
                results['success'] = True
                results['approach'] = 'direct_answer'
                logger.info("Intelligent research completed with direct answer (no web search needed)")
                return results
            
            # If web search is needed, continue with the full process
            search_query = necessity_check['search_query']
            results['approach'] = 'web_search_research'
            results['steps']['step0']['search_query'] = search_query
            
            # Step 2: Web search
            search_results = await self.step2_web_search(search_query)
            results['steps']['step2'] = {
                'description': 'Perform web search',
                'search_results': search_results,
                'results_count': len(search_results),
                'success': len(search_results) > 0
            }
            
            if not search_results:
                results['error'] = 'No search results found'
                return results
            
            # Step 3: Extract content
            extracted_contents = await self.step3_extract_content(search_results)
            results['steps']['step3'] = {
                'description': 'Extract content from web pages',
                'extracted_contents': extracted_contents,
                'successful_extractions': sum(1 for c in extracted_contents if c['success']),
                'total_extractions': len(extracted_contents),
                'success': any(c['success'] for c in extracted_contents)
            }
            
            # Step 4: Analyze relevance
            analysis = await self.step4_analyze_relevance(original_question, extracted_contents)
            results['steps']['step4'] = {
                'description': 'Analyze content relevance with DeepSeek Reasoning',
                'analysis': analysis,
                'success': 'error' not in analysis
            }
            
            results['success'] = True
            logger.info("Intelligent research completed successfully")
            
        except Exception as e:
            logger.error(f"Intelligent research failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results

    async def conduct_legacy_research(self, original_question: str) -> Dict[str, Any]:
        """Conduct legacy 4-step research process (always performs web search)"""
        logger.info(f"Starting legacy research for: {original_question}")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'research_type': 'legacy',
            'approach': 'legacy_web_search',
            'steps': {}
        }
        
        try:
            # Step 1: Generate search query
            search_query = await self.step1_generate_search_query(original_question)
            results['steps']['step1'] = {
                'description': 'Generate optimal search query',
                'search_query': search_query,
                'success': True
            }
            
            # Step 2: Web search
            search_results = await self.step2_web_search(search_query)
            results['steps']['step2'] = {
                'description': 'Perform web search',
                'search_results': search_results,
                'results_count': len(search_results),
                'success': len(search_results) > 0
            }
            
            if not search_results:
                results['error'] = 'No search results found'
                return results
            
            # Step 3: Extract content
            extracted_contents = await self.step3_extract_content(search_results)
            results['steps']['step3'] = {
                'description': 'Extract content from web pages',
                'extracted_contents': extracted_contents,
                'successful_extractions': sum(1 for c in extracted_contents if c['success']),
                'total_extractions': len(extracted_contents),
                'success': any(c['success'] for c in extracted_contents)
            }
            
            # Step 4: Analyze relevance
            analysis = await self.step4_analyze_relevance(original_question, extracted_contents)
            results['steps']['step4'] = {
                'description': 'Analyze content relevance with DeepSeek Reasoning',
                'analysis': analysis,
                'success': 'error' not in analysis
            }
            
            results['success'] = True
            logger.info("Legacy research completed successfully")
            
        except Exception as e:
            logger.error(f"Legacy research failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results

    async def conduct_dual_research(self, original_question: str) -> Dict[str, Any]:
        """
        Conduct both intelligent and legacy research approaches simultaneously
        
        Args:
            original_question: The original research question
            
        Returns:
            Complete dual research results with both approaches
        """
        logger.info(f"Starting dual research for: {original_question}")
        
        self._notify_progress("dual_start", {"question": original_question})
        
        dual_results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'research_type': 'dual',
            'intelligent_research': {},
            'legacy_research': {}
        }
        
        try:
            # Run both research approaches concurrently
            self._notify_progress("dual_running", {"description": "Running both intelligent and legacy research"})
            
            intelligent_task = asyncio.create_task(self.conduct_intelligent_research(original_question))
            legacy_task = asyncio.create_task(self.conduct_legacy_research(original_question))
            
            # Wait for both to complete
            intelligent_result, legacy_result = await asyncio.gather(intelligent_task, legacy_task)
            
            dual_results['intelligent_research'] = intelligent_result
            dual_results['legacy_research'] = legacy_result
            
            # Determine overall success
            dual_results['success'] = (
                intelligent_result.get('success', False) or 
                legacy_result.get('success', False)
            )
            
            logger.info("Dual research completed successfully")
            self._notify_progress("dual_complete", dual_results)
            
        except Exception as e:
            logger.error(f"Dual research failed: {e}")
            dual_results['error'] = str(e)
            dual_results['success'] = False
            self._notify_progress("dual_error", dual_results)
        
        return dual_results