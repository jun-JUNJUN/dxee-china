#!/usr/bin/env python3
"""
Advanced DeepSeek Web Research with Multi-Step Process
This script implements a sophisticated research workflow:
1. Use DeepSeek to generate optimal search queries
2. Perform Google web search
3. Extract full article content from search results
4. Use DeepSeek Reasoning to analyze and verify relevance

Usage:
    python test_deepseek_advanced_web_research.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: The base URL for the DeepSeek API (default: https://api.deepseek.com)
    GOOGLE_API_KEY: Your Google Custom Search API key
    GOOGLE_CSE_ID: Your Google Custom Search Engine ID
"""

import os
import sys
import json
import asyncio
import logging
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import time

# Web scraping libraries
try:
    from bs4 import BeautifulSoup
    import newspaper
    from newspaper import Article
    from readability import Document
except ImportError as e:
    print(f"? Missing required libraries: {e}")
    print("? Please install: pip install beautifulsoup4 lxml newspaper3k readability-lxml")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deepseek_advanced_research.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
    
    async def search(self, query: str, num_results: int = 5, start_index: int = 1) -> List[Dict[str, Any]]:
        """
        Perform a Google web search
        
        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            start_index: Start index for pagination (1-based)
            
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
                'start': start_index,
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

class DeepSeekAdvancedResearchService:
    """Advanced research service with multi-step process"""
    
    def __init__(self):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.web_search = GoogleWebSearchService()
        self.content_extractor = WebContentExtractor()
        
        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    async def step0_check_web_search_necessity(self, original_question: str) -> Dict[str, Any]:
        """
        Step 0: Check if web search is necessary for answering the question
        
        Args:
            original_question: The original research question
            
        Returns:
            Dictionary with decision and either search query or direct answer
        """
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
        """
        Step 1: Use DeepSeek to generate optimal search query (legacy method)
        
        Args:
            original_question: The original research question
            
        Returns:
            Optimized search query in format Query="BBB"
        """
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
    
    async def step2_web_search(self, search_query: str, start_index: int = 1) -> List[Dict[str, Any]]:
        """
        Step 2: Perform Google web search
        
        Args:
            search_query: The optimized search query
            start_index: Start index for pagination (1-based)
            
        Returns:
            List of search results
        """
        logger.info(f"Step 2: Performing web search (start index: {start_index})")
        return await self.web_search.search(search_query, num_results=5, start_index=start_index)
    
    async def step3_extract_content(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: Extract full content from search result pages
        
        Args:
            search_results: List of search results from Google
            
        Returns:
            List of extracted content with metadata
        """
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
        """
        Step 4: Use DeepSeek Reasoning to analyze content relevance
        
        Args:
            original_question: The original research question
            extracted_contents: List of extracted content from web pages
            
        Returns:
            Analysis results with relevance assessment
        """
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
7. Provide your reasoning process clearly
8. IMPORTANT: At the very end of your response, provide an OVERALL RELEVANCE SCORE (1-10) in the format: "OVERALL_RELEVANCE_SCORE: X" where X is your assessment of how well all sources combined answer the original question"""

            user_prompt = f"""Original Research Question: {original_question}

Extracted Content from Web Sources:
{''.join(content_summaries)}

Please analyze each source for relevance to the original question and provide:
1. Individual source analysis with relevance scores
2. Key findings that answer the original question
3. Synthesis of information across sources
4. Analysis of any discrepancies or conflicting information
5. Overall assessment of how well the sources answer the original question"""

            print("\n? Starting DeepSeek Reasoning Analysis (Live Stream)...")
            print("? [REASONING] Starting analysis...", flush=True)
            
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                timeout=60.0
            )
            
            # Initialize variables for streaming
            reasoning_content = ""
            analysis_content = ""
            reasoning_buffer = ""
            analysis_buffer = ""
            
            # Process streaming response
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle reasoning content streaming
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_chunk = delta.reasoning_content
                        reasoning_content += reasoning_chunk
                        reasoning_buffer += reasoning_chunk
                        
                        # Display reasoning content as it streams
                        if len(reasoning_buffer) > 50:  # Display in chunks to avoid too frequent updates
                            print(f"\n? [REASONING] ...{reasoning_buffer[-50:]}", end="", flush=True)
                            reasoning_buffer = ""
                    
                    # Handle regular content streaming
                    if hasattr(delta, 'content') and delta.content:
                        content_chunk = delta.content
                        analysis_content += content_chunk
                        analysis_buffer += content_chunk
                        
                        # Display analysis content as it streams
                        if len(analysis_buffer) > 50:  # Display in chunks to avoid too frequent updates
                            print(f"\n? [ANALYSIS] ...{analysis_buffer[-50:]}", end="", flush=True)
                            analysis_buffer = ""
            
            # Display any remaining content
            if reasoning_buffer:
                print(f"\n? [REASONING] ...{reasoning_buffer}", end="", flush=True)
            if analysis_buffer:
                print(f"\n? [ANALYSIS] ...{analysis_buffer}", end="", flush=True)
            
            print(f"\n? Analysis streaming completed!")
            
            # Extract overall relevance score from analysis content
            overall_relevance_score = 0
            score_match = re.search(r'OVERALL_RELEVANCE_SCORE:\s*(\d+)', analysis_content)
            if score_match:
                overall_relevance_score = int(score_match.group(1))
                logger.info(f"Extracted overall relevance score: {overall_relevance_score}")
                print(f"? Extracted relevance score: {overall_relevance_score}/10")
            else:
                logger.warning("Could not extract overall relevance score from analysis")
                print("? Warning: Could not extract relevance score from analysis")
            
            return {
                'original_question': original_question,
                'analysis_content': analysis_content,
                'reasoning_content': reasoning_content,
                'overall_relevance_score': overall_relevance_score,
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
    
    async def conduct_advanced_research(self, original_question: str) -> Dict[str, Any]:
        """
        Conduct the complete advanced research process with web search necessity check
        
        Args:
            original_question: The original research question
            
        Returns:
            Complete research results
        """
        logger.info(f"Starting advanced research for: {original_question}")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
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
                results['research_type'] = 'direct_answer'
                logger.info("Research completed with direct answer (no web search needed)")
                return results
            
            # If web search is needed, continue with the full process
            search_query = necessity_check['search_query']
            results['research_type'] = 'web_search_research'
            results['steps']['step0']['search_query'] = search_query
            
            # Initialize variables for iterative search
            all_search_results = []
            all_extracted_contents = []
            search_iteration = 0
            max_iterations = 4  # Up to 4 iterations to get 20 total results (5 per iteration)
            relevance_threshold = 8
            current_relevance_score = 0
            
            # Iterative search loop
            while search_iteration < max_iterations and len(all_search_results) < 20:
                search_iteration += 1
                start_index = len(all_search_results) + 1  # Google uses 1-based indexing
                
                logger.info(f"Search iteration {search_iteration}, starting from result {start_index}")
                
                # Step 2: Web search
                search_results = await self.step2_web_search(search_query, start_index)
                if not search_results:
                    logger.warning(f"No search results found in iteration {search_iteration}")
                    break
                
                all_search_results.extend(search_results)
                
                # Step 3: Extract content for new results
                extracted_contents = await self.step3_extract_content(search_results)
                all_extracted_contents.extend(extracted_contents)
                
                # Step 4: Analyze relevance of all content so far
                analysis = await self.step4_analyze_relevance(original_question, all_extracted_contents)
                current_relevance_score = analysis.get('overall_relevance_score', 0)
                
                logger.info(f"Iteration {search_iteration}: Relevance score = {current_relevance_score}")
                
                # Check if relevance threshold is met
                if current_relevance_score >= relevance_threshold:
                    logger.info(f"Relevance threshold {relevance_threshold} met with score {current_relevance_score}")
                    break
                
                # If not meeting threshold and haven't reached max results, continue
                if len(all_search_results) < 20:
                    logger.info(f"Relevance score {current_relevance_score} < {relevance_threshold}, continuing search...")
            
            # Store final results
            results['steps']['step2'] = {
                'description': 'Perform iterative web search',
                'search_results': all_search_results,
                'results_count': len(all_search_results),
                'search_iterations': search_iteration,
                'success': len(all_search_results) > 0
            }
            
            results['steps']['step3'] = {
                'description': 'Extract content from web pages',
                'extracted_contents': all_extracted_contents,
                'successful_extractions': sum(1 for c in all_extracted_contents if c['success']),
                'total_extractions': len(all_extracted_contents),
                'success': any(c['success'] for c in all_extracted_contents)
            }
            
            # Final analysis (this will be the last analysis from the loop)
            if all_extracted_contents:
                final_analysis = await self.step4_analyze_relevance(original_question, all_extracted_contents)
                results['steps']['step4'] = {
                    'description': 'Analyze content relevance with DeepSeek Reasoning',
                    'analysis': final_analysis,
                    'final_relevance_score': final_analysis.get('overall_relevance_score', 0),
                    'success': 'error' not in final_analysis
                }
            else:
                results['error'] = 'No content extracted from any search results'
                return results
            
            results['success'] = True
            logger.info("Advanced research completed successfully")
            
        except Exception as e:
            logger.error(f"Advanced research failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results

    async def conduct_legacy_research(self, original_question: str) -> Dict[str, Any]:
        """
        Conduct the legacy 4-step research process (always performs web search)
        
        Args:
            original_question: The original research question
            
        Returns:
            Complete research results
        """
        logger.info(f"Starting legacy research for: {original_question}")
        
        results = {
            'original_question': original_question,
            'timestamp': datetime.utcnow().isoformat(),
            'steps': {},
            'research_type': 'legacy_web_search'
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

def print_separator(char="=", length=80):
    """Print a separator line"""
    print("\n" + char * length + "\n")

def print_step_header(step_num: int, description: str):
    """Print a step header"""
    print(f"\n? STEP {step_num}: {description}")
    print("-" * 60)

def print_streaming_content(content_type: str, text: str):
    """Print streaming content with proper formatting"""
    if content_type == "reasoning":
        print(f"\r? [REASONING] {text}", end="", flush=True)
    elif content_type == "answer":
        print(f"\r? [ANALYSIS] {text}", end="", flush=True)
    else:
        print(f"\r? {text}", end="", flush=True)

async def test_advanced_research():
    """Test the advanced research process"""
    print_separator()
    print("? DEEPSEEK ADVANCED WEB RESEARCH TEST")
    print("Multi-Step Research Process:")
    print("1. Generate optimal search query")
    print("2. Perform web search")
    print("3. Extract full article content")
    print("4. Analyze relevance with reasoning")
    print_separator()
    
    # Initialize service
    try:
        service = DeepSeekAdvancedResearchService()
    except ValueError as e:
        print(f"? {e}")
        return
    
    # Test question
    original_question = "Find the CRM/SFA software available in Japan and make the rank by their revenues"
    print(f"? Research Question: {original_question}")
    
    # Conduct research
    results = await service.conduct_advanced_research(original_question)
    
    if not results.get('success'):
        print(f"? Research failed: {results.get('error', 'Unknown error')}")
        return
    
    # Display results
    print_step_header(1, "Search Query Generation")
    step1 = results['steps'].get('step1', {})
    if step1.get('success'):
        print(f"? Generated search query: \"{step1['search_query']}\"")
    else:
        print("? Failed to generate search query")
    
    print_step_header(2, "Web Search Results")
    step2 = results['steps'].get('step2', {})
    if step2.get('success'):
        print(f"? Found {step2['results_count']} search results")
        for i, result in enumerate(step2['search_results'][:3], 1):
            print(f"  {i}. {result['title']}")
            print(f"     URL: {result['link']}")
            print(f"     Snippet: {result['snippet'][:100]}...")
    else:
        print("? No search results found")
    
    print_step_header(3, "Content Extraction")
    step3 = results['steps'].get('step3', {})
    if step3.get('success'):
        print(f"? Successfully extracted content from {step3['successful_extractions']}/{step3['total_extractions']} pages")
        for i, content in enumerate(step3['extracted_contents'], 1):
            status = "?" if content['success'] else "?"
            print(f"  {status} Source {i}: {content['title']}")
            print(f"     Method: {content['method']}, Words: {content['word_count']}")
            if not content['success']:
                print(f"     Error: {content.get('error', 'Unknown error')}")
    else:
        print("? Failed to extract content from any pages")
    
    print_step_header(4, "Relevance Analysis")
    step4 = results['steps'].get('step4', {})
    if step4.get('success'):
        analysis = step4['analysis']
        print(f"? Analysis completed using {analysis.get('model', 'unknown')} model")
        print(f"? Sources analyzed: {analysis.get('sources_analyzed', 0)}")
        print(f"? Successful extractions: {analysis.get('successful_extractions', 0)}")
        
        if analysis.get('reasoning_content'):
            print("\n? REASONING PROCESS:")
            reasoning = analysis['reasoning_content']
            print(reasoning[:1000] + "..." if len(reasoning) > 1000 else reasoning)
        
        print("\n? FINAL ANALYSIS:")
        print(analysis.get('analysis_content', 'No analysis available'))
    else:
        print("? Analysis failed")
        if 'analysis' in step4 and 'error' in step4['analysis']:
            print(f"Error: {step4['analysis']['error']}")
    
    print_separator()
    print("? Advanced research process completed!")
async def test_intelligent_research():
    """Test the intelligent research process with web search necessity check"""
    print_separator()
    print("? DEEPSEEK INTELLIGENT WEB RESEARCH TEST")
    print("Intelligent Research Process:")
    print("0. Check if web search is necessary")
    print("1. If needed: Generate optimal search query")
    print("2. If needed: Perform iterative web search (up to 20 results)")
    print("3. If needed: Extract full article content")
    print("4. If needed: Analyze relevance with reasoning")
    print("   - Continue search iterations until relevance ≥ 8 or max 20 results")
    print_separator()
    
    # Initialize service
    try:
        service = DeepSeekAdvancedResearchService()
    except ValueError as e:
        print(f"? {e}")
        return
    
    # Test questions - one that needs web search, one that doesn't
    test_questions = [
        "Find the CRM/SFA software available in Japan and make the rank by their revenues",
        "What is the difference between a list and a tuple in Python programming?"
    ]
    
    for i, original_question in enumerate(test_questions, 1):
        print(f"\n? TEST {i}: {original_question}")
        print("-" * 80)
        
        # Conduct research
        results = await service.conduct_advanced_research(original_question)
        
        if not results.get('success'):
            print(f"? Research failed: {results.get('error', 'Unknown error')}")
            continue
        
        # Display results based on research type
        print_step_header(0, "Web Search Necessity Check")
        step0 = results['steps'].get('step0', {})
        if step0.get('success'):
            web_search_needed = step0['web_search_needed']
            if web_search_needed:
                print(f"? Web search NEEDED")
                print(f"? Generated search query: \"{step0.get('search_query', 'N/A')}\"")
            else:
                print(f"? Web search NOT needed - providing direct answer")
        
        if results.get('research_type') == 'direct_answer':
            print("\n? DIRECT ANSWER:")
            print(results.get('direct_answer', 'No answer available'))
            print("\n? Research completed without web search!")
            continue
        
        # If web search was performed, show those results
        print_step_header(2, "Iterative Web Search Results")
        step2 = results['steps'].get('step2', {})
        if step2.get('success'):
            iterations = step2.get('search_iterations', 1)
            print(f"? Completed {iterations} search iteration(s)")
            print(f"? Found {step2['results_count']} total search results")
            for j, result in enumerate(step2['search_results'][:3], 1):
                print(f"  {j}. {result['title']}")
                print(f"     URL: {result['link']}")
            if len(step2['search_results']) > 3:
                print(f"  ... and {len(step2['search_results']) - 3} more results")
        else:
            print("? No search results found")
        
        print_step_header(3, "Content Extraction")
        step3 = results['steps'].get('step3', {})
        if step3.get('success'):
            print(f"? Successfully extracted content from {step3['successful_extractions']}/{step3['total_extractions']} pages")
        else:
            print("? Failed to extract content from any pages")
        
        print_step_header(4, "Relevance Analysis")
        step4 = results['steps'].get('step4', {})
        if step4.get('success'):
            analysis = step4['analysis']
            final_score = step4.get('final_relevance_score', 0)
            print(f"? Analysis completed using {analysis.get('model', 'unknown')} model")
            print(f"? Final relevance score: {final_score}/10")
            if final_score >= 8:
                print("? ✓ Relevance threshold (8) achieved!")
            else:
                print(f"? ⚠ Relevance threshold (8) not achieved, final score: {final_score}")
            print("\n? FINAL ANALYSIS:")
            analysis_content = analysis.get('analysis_content', 'No analysis available')
            print(analysis_content[:800] + "..." if len(analysis_content) > 800 else analysis_content)
        else:
            print("? Analysis failed")
        
        print(f"\n? Test {i} completed!")
    
    print_separator()
    print("? All intelligent research tests completed!")

# async def test_legacy_research():
#     """Test the legacy research process (always performs web search)"""
#     print_separator()
#     print("? DEEPSEEK LEGACY WEB RESEARCH TEST")
#     print("Legacy Research Process (Always Web Search):")
#     print("1. Generate optimal search query")
#     print("2. Perform web search")
#     print("3. Extract full article content")
#     print("4. Analyze relevance with reasoning")
#     print_separator()
#     
#     # Initialize service
#     try:
#         service = DeepSeekAdvancedResearchService()
#     except ValueError as e:
#         print(f"? {e}")
#         return
#     
#     # Test question that normally wouldn't need web search
#     original_question = "What is the difference between a list and a tuple in Python programming?"
#     print(f"? Research Question: {original_question}")
#     print("(This question normally wouldn't need web search, but legacy mode forces it)")
#     
#     # Conduct legacy research
#     results = await service.conduct_legacy_research(original_question)
#     
#     if not results.get('success'):
#         print(f"? Research failed: {results.get('error', 'Unknown error')}")
#         return
#     
#     # Display results
#     print_step_header(1, "Search Query Generation")
#     step1 = results['steps'].get('step1', {})
#     if step1.get('success'):
#         print(f"? Generated search query: \"{step1['search_query']}\"")
#     else:
#         print("? Failed to generate search query")
#     
#     print_step_header(2, "Web Search Results")
#     step2 = results['steps'].get('step2', {})
#     if step2.get('success'):
#         print(f"? Found {step2['results_count']} search results")
#         for i, result in enumerate(step2['search_results'][:2], 1):
#             print(f"  {i}. {result['title']}")
#             print(f"     URL: {result['link']}")
#     else:
#         print("? No search results found")
#     
#     print_step_header(3, "Content Extraction")
#     step3 = results['steps'].get('step3', {})
#     if step3.get('success'):
#         print(f"? Successfully extracted content from {step3['successful_extractions']}/{step3['total_extractions']} pages")
#     else:
#         print("? Failed to extract content from any pages")
#     
#     print_step_header(4, "Relevance Analysis")
#     step4 = results['steps'].get('step4', {})
#     if step4.get('success'):
#         analysis = step4['analysis']
#         print(f"? Analysis completed using {analysis.get('model', 'unknown')} model")
#         print("\n? FINAL ANALYSIS:")
#         analysis_content = analysis.get('analysis_content', 'No analysis available')
#         print(analysis_content[:800] + "..." if len(analysis_content) > 800 else analysis_content)
#     else:
#         print("? Analysis failed")
#     
#     print_separator()
#     print("? Legacy research process completed!")

def check_environment():
    """Check if required environment variables are set"""
    print("? Checking environment variables...")
    
    required_vars = {
        'DEEPSEEK_API_KEY': os.environ.get('DEEPSEEK_API_KEY'),
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY'),
        'GOOGLE_CSE_ID': os.environ.get('GOOGLE_CSE_ID')
    }
    
    missing_vars = []
    for var, value in required_vars.items():
        if value and value != f"your_{var.lower()}_here":
            masked_value = f"{value[:5]}...{value[-5:]}" if len(value) > 10 else "***"
            print(f"? {var}: {masked_value}")
        else:
            print(f"? {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n??  Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease configure your .env file with the required API keys.")
        return False
    
    return True

async def main():
    """Main test function"""
    print("DEEPSEEK ADVANCED WEB RESEARCH")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n? Environment setup incomplete. Please configure required variables.")
        sys.exit(1)
    
    try:
        # Run the intelligent research test (with web search necessity check)
        await test_intelligent_research()
        
        # Optionally run legacy test to show the difference
        # print("\n" + "=" * 80)
        # print("COMPARISON: Legacy Research Mode")
        # print("=" * 80)
        # await test_legacy_research()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n? Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
