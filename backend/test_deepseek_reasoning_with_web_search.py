#!/usr/bin/env python3
"""
Test script for DeepSeek API with Reasoning model integrated with Google Web Search.
This script demonstrates how to use DeepSeek's reasoning capabilities with real-time web search.

Usage:
    python test_deepseek_reasoning_with_web_search.py

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
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deepseek_web_search_test.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GoogleWebSearchService:
    """Service for performing Google web searches using Custom Search API"""
    
    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. Web search functionality will be limited.")
        if not self.cse_id:
            logger.warning("GOOGLE_CSE_ID not set. Web search functionality will be limited.")
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
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

class DeepSeekReasoningWithWebSearch:
    """DeepSeek service with integrated web search capabilities"""
    
    def __init__(self):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        self.api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.web_search = GoogleWebSearchService()
        
        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for inclusion in the prompt"""
        if not results:
            return "No web search results available."
        
        formatted = "Web Search Results:\n"
        for i, result in enumerate(results, 1):
            formatted += f"\n{i}. {result['title']}\n"
            formatted += f"   URL: {result['link']}\n"
            formatted += f"   Summary: {result['snippet']}\n"
        
        return formatted
    
    async def reasoning_with_web_search(self, query: str, search_query: str = None) -> Dict[str, Any]:
        """
        Perform reasoning with web search integration
        
        Args:
            query: The main query for reasoning
            search_query: Optional specific search query (defaults to main query)
            
        Returns:
            Dictionary containing reasoning, answer, and search results
        """
        try:
            # Use the main query for search if no specific search query provided
            if search_query is None:
                search_query = query
            
            logger.info(f"Starting reasoning with web search for: {query}")
            
            # Perform web search first
            search_results = await self.web_search.search(search_query, num_results=8)
            formatted_search_results = self._format_search_results(search_results)
            
            # Create enhanced system message
            system_message = """You are an expert research assistant with access to current web information. 
You have been provided with recent web search results to help answer the user's question accurately.

Instructions:
1. Analyze the provided web search results carefully
2. Use your reasoning capabilities to synthesize information from multiple sources
3. Provide accurate, up-to-date information based on the search results
4. If the search results don't contain sufficient information, clearly state this
5. Always cite your sources when possible
6. Be thorough in your analysis and reasoning process"""

            # Create enhanced user prompt with search results
            enhanced_query = f"""User Query: {query}

{formatted_search_results}

Please analyze the above web search results and provide a comprehensive answer to the user's query. Use your reasoning capabilities to synthesize the information and provide insights."""

            logger.info("Sending request to DeepSeek Reasoner model")
            
            # Use deepseek-reasoner model for enhanced reasoning
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": enhanced_query}
                ],
                stream=False,
                timeout=60.0
            )
            
            logger.info("DeepSeek reasoning completed successfully")
            
            # Extract reasoning and answer content
            message = response.choices[0].message
            reasoning_content = getattr(message, 'reasoning_content', '') if hasattr(message, 'reasoning_content') else ''
            answer_content = message.content or ''
            
            return {
                'query': query,
                'search_query': search_query,
                'search_results': search_results,
                'reasoning_content': reasoning_content,
                'answer_content': answer_content,
                'model': response.model,
                'timestamp': datetime.utcnow().isoformat(),
                'search_results_count': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Reasoning with web search failed: {e}")
            return {
                'query': query,
                'search_query': search_query,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def streaming_reasoning_with_web_search(self, query: str, search_query: str = None):
        """
        Perform streaming reasoning with web search integration
        
        Args:
            query: The main query for reasoning
            search_query: Optional specific search query (defaults to main query)
            
        Yields:
            Streaming chunks with reasoning and answer content
        """
        try:
            # Use the main query for search if no specific search query provided
            if search_query is None:
                search_query = query
            
            logger.info(f"Starting streaming reasoning with web search for: {query}")
            
            # Perform web search first
            search_results = await self.web_search.search(search_query, num_results=8)
            formatted_search_results = self._format_search_results(search_results)
            
            # Yield search results first
            yield {
                'type': 'search_results',
                'search_results': search_results,
                'search_results_count': len(search_results),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Create enhanced system message
            system_message = """You are an expert research assistant with access to current web information. 
You have been provided with recent web search results to help answer the user's question accurately.

Instructions:
1. Analyze the provided web search results carefully
2. Use your reasoning capabilities to synthesize information from multiple sources
3. Provide accurate, up-to-date information based on the search results
4. If the search results don't contain sufficient information, clearly state this
5. Always cite your sources when possible
6. Be thorough in your analysis and reasoning process"""

            # Create enhanced user prompt with search results
            enhanced_query = f"""User Query: {query}

{formatted_search_results}

Please analyze the above web search results and provide a comprehensive answer to the user's query. Use your reasoning capabilities to synthesize the information and provide insights."""

            logger.info("Starting streaming request to DeepSeek Reasoner model")
            
            # Use deepseek-reasoner model for enhanced reasoning with streaming
            stream = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": enhanced_query}
                ],
                stream=True,
                timeout=60.0
            )
            
            reasoning_content = ""
            answer_content = ""
            
            async for chunk in stream:
                # Handle reasoning content
                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    reasoning_chunk = chunk.choices[0].delta.reasoning_content
                    reasoning_content += reasoning_chunk
                    yield {
                        'type': 'reasoning_chunk',
                        'reasoning_content': reasoning_chunk,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                
                # Handle answer content
                elif chunk.choices[0].delta.content:
                    answer_chunk = chunk.choices[0].delta.content
                    answer_content += answer_chunk
                    yield {
                        'type': 'answer_chunk',
                        'content': answer_chunk,
                        'timestamp': datetime.utcnow().isoformat()
                    }
            
            # Send completion signal
            yield {
                'type': 'complete',
                'query': query,
                'search_query': search_query,
                'reasoning_content': reasoning_content,
                'answer_content': answer_content,
                'search_results': search_results,
                'search_results_count': len(search_results),
                'reasoning_length': len(reasoning_content),
                'answer_length': len(answer_content),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Streaming reasoning with web search failed: {e}")
            yield {
                'type': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

def print_separator(char="=", length=80):
    """Print a separator line"""
    print("\n" + char * length + "\n")

def print_search_results(results: List[Dict[str, Any]]):
    """Print search results in a formatted way"""
    if not results:
        print("No search results found.")
        return
    
    print(f"Found {len(results)} search results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['link']}")
        print(f"   Summary: {result['snippet'][:200]}...")

async def test_basic_reasoning_with_search():
    """Test basic reasoning with web search"""
    print_separator()
    print("TEST 1: Basic Reasoning with Web Search")
    print_separator("-")
    
    service = DeepSeekReasoningWithWebSearch()
    
    # Test query about CRM/SFA software in Japan
    query = "Find the CRM/SFA software available in Japan and make the rank by their revenues"
    
    print(f"Query: {query}")
    print("\nProcessing...")
    
    result = await service.reasoning_with_web_search(query)
    
    if 'error' in result:
        print(f"? Error: {result['error']}")
        return
    
    print(f"? Completed successfully!")
    print(f"Model: {result.get('model', 'Unknown')}")
    print(f"Search Results: {result.get('search_results_count', 0)} found")
    
    # Print search results
    print_separator("-")
    print("SEARCH RESULTS:")
    print_search_results(result.get('search_results', []))
    
    # Print reasoning if available
    if result.get('reasoning_content'):
        print_separator("-")
        print("REASONING PROCESS:")
        print(result['reasoning_content'][:1000] + "..." if len(result['reasoning_content']) > 1000 else result['reasoning_content'])
    
    # Print answer
    print_separator("-")
    print("FINAL ANSWER:")
    print(result.get('answer_content', 'No answer provided'))

async def test_streaming_reasoning_with_search():
    """Test streaming reasoning with web search"""
    print_separator()
    print("TEST 2: Streaming Reasoning with Web Search")
    print_separator("-")
    
    service = DeepSeekReasoningWithWebSearch()
    
    # Test query about CRM/SFA software in Japan
    query = "Find the CRM/SFA software available in Japan and make the rank by their revenues"
    
    print(f"Query: {query}")
    print("\nStreaming results...")
    
    search_results = []
    reasoning_content = ""
    answer_content = ""
    
    async for chunk in service.streaming_reasoning_with_web_search(query):
        if chunk['type'] == 'search_results':
            search_results = chunk['search_results']
            print(f"\n? Found {chunk['search_results_count']} search results")
            
        elif chunk['type'] == 'reasoning_chunk':
            reasoning_content += chunk['reasoning_content']
            print("?", end="", flush=True)  # Show reasoning progress
            
        elif chunk['type'] == 'answer_chunk':
            answer_content += chunk['content']
            print(chunk['content'], end="", flush=True)
            
        elif chunk['type'] == 'complete':
            print(f"\n\n? Streaming completed!")
            print(f"Total reasoning length: {chunk['reasoning_length']} characters")
            print(f"Total answer length: {chunk['answer_length']} characters")
            break
            
        elif chunk['type'] == 'error':
            print(f"\n? Error: {chunk['error']}")
            return
    
    # Print final results
    print_separator("-")
    print("SEARCH RESULTS:")
    print_search_results(search_results)
    
    if reasoning_content:
        print_separator("-")
        print("REASONING PROCESS:")
        print(reasoning_content[:1000] + "..." if len(reasoning_content) > 1000 else reasoning_content)

def check_environment():
    """Check if required environment variables are set"""
    print("Checking environment variables...")
    
    required_vars = {
        'DEEPSEEK_API_KEY': os.environ.get('DEEPSEEK_API_KEY'),
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY'),
        'GOOGLE_CSE_ID': os.environ.get('GOOGLE_CSE_ID')
    }
    
    missing_vars = []
    for var, value in required_vars.items():
        if value:
            masked_value = f"{value[:5]}...{value[-5:]}" if len(value) > 10 else "***"
            print(f"? {var}: {masked_value}")
        else:
            print(f"? {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n??  Missing environment variables: {', '.join(missing_vars)}")
        print("\nTo set up Google Custom Search API:")
        print("1. Go to https://console.developers.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable Custom Search API")
        print("4. Create credentials (API key)")
        print("5. Go to https://cse.google.com/cse/")
        print("6. Create a custom search engine")
        print("7. Get the Search Engine ID")
        print("\nAdd to your .env file:")
        print("GOOGLE_API_KEY=your_google_api_key_here")
        print("GOOGLE_CSE_ID=your_custom_search_engine_id_here")
        return False
    
    return True

async def main():
    """Main test function"""
    print("DEEPSEEK REASONING WITH WEB SEARCH TEST")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n? Environment setup incomplete. Please configure required variables.")
        sys.exit(1)
    
    try:
        # Run tests
        await test_basic_reasoning_with_search()
        await test_streaming_reasoning_with_search()
        
        print_separator()
        print("? All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n? Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
