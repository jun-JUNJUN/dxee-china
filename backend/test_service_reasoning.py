#!/usr/bin/env python3
"""
Test script using the DeepSeekService class to demonstrate reasoning and answer content.
This script tests the specific query about saying 'hello' in different languages
using the existing service infrastructure.

Usage:
    python test_service_reasoning.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: The base URL for the DeepSeek API (default: https://api.deepseek.com)
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.service.deepseek_service import DeepSeekService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('service_reasoning_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")

def print_section(title, content, prefix=""):
    """Print a formatted section with content."""
    print(f"\n{prefix}┌─ {title}")
    lines = content.split('\n')
    for line in lines:
        print(f"{prefix}│ {line}")
    print(f"{prefix}└─")

async def test_streaming_with_reasoning():
    """
    Test the DeepSeek service streaming functionality to capture both reasoning and answer.
    """
    # The specific query requested
    query = "How many languages do you say 'hello'? Please show me 'hello' in each language you can."
    
    print_header("DEEPSEEK SERVICE REASONING TEST")
    print(f"Query: {query}")
    
    # Initialize the service (using empty queues since we're testing directly)
    input_queue = []
    output_queue = []
    service = DeepSeekService(input_queue, output_queue)
    
    # Check if API key is configured
    if not service.api_key:
        print("❌ DEEPSEEK_API_KEY is not configured!")
        return False
    
    print(f"✅ API Key configured: {service.api_key[:8]}...{service.api_key[-4:]}")
    print(f"🌐 API URL: {service.api_url}")
    
    try:
        print("\n🚀 Testing streaming with reasoning content...")
        
        # Create a simple queue to collect streaming results
        stream_queue = asyncio.Queue()
        
        # Prepare message data for the service
        message_data = {
            'message': query,
            'chat_id': 'test_chat_001',
            'message_id': 'test_msg_001',
            'search_mode': 'deep',  # Use 'deep' mode to prefer deepseek-reasoner
            'streaming': True,
            'chat_history': []
        }
        
        print("📡 Starting streaming request...")
        
        # Process the message with streaming
        await service.process_message_stream(message_data, stream_queue)
        
        # Collect all streaming chunks
        reasoning_chunks = []
        answer_chunks = []
        complete_reasoning = ""
        complete_answer = ""
        
        print("\n📥 Collecting streaming chunks...")
        
        while True:
            try:
                # Wait for chunks with timeout
                chunk = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                
                chunk_type = chunk.get('type', 'unknown')
                print(f"📦 Received chunk type: {chunk_type}")
                
                if chunk_type == 'reasoning_chunk':
                    reasoning_content = chunk.get('reasoning_content', '')
                    reasoning_chunks.append(reasoning_content)
                    print(f"🧠 Reasoning: {reasoning_content[:50]}{'...' if len(reasoning_content) > 50 else ''}")
                
                elif chunk_type == 'chunk':
                    answer_content = chunk.get('content', '')
                    answer_chunks.append(answer_content)
                    print(f"💬 Answer: {answer_content[:50]}{'...' if len(answer_content) > 50 else ''}")
                
                elif chunk_type == 'complete':
                    complete_reasoning = chunk.get('reasoning_content', '')
                    complete_answer = chunk.get('content', '')
                    formatted_reasoning = chunk.get('formatted_reasoning', '')
                    formatted_answer = chunk.get('formatted_content', '')
                    
                    print("✅ Received complete response!")
                    break
                
                elif chunk_type == 'error':
                    error_content = chunk.get('content', 'Unknown error')
                    print(f"❌ Error: {error_content}")
                    return False
                
            except asyncio.TimeoutError:
                print("⏰ No more chunks received (timeout)")
                break
        
        # Display results
        print_header("COMPLETE RESULTS")
        
        if complete_reasoning:
            print_section("🧠 REASONING CONTENT (Thinking Process)", complete_reasoning, "🔵 ")
            print(f"📊 Reasoning length: {len(complete_reasoning)} characters")
        else:
            print("⚠️  No reasoning content received")
        
        if complete_answer:
            print_section("💬 ANSWER CONTENT (Final Response)", complete_answer, "🟢 ")
            print(f"📊 Answer length: {len(complete_answer)} characters")
        else:
            print("❌ No answer content received")
        
        # Analyze the content
        print_header("CONTENT ANALYSIS")
        
        # Count potential languages in the answer
        common_hellos = [
            "hello", "hi", "hola", "bonjour", "guten tag", "ciao", "konnichiwa",
            "ni hao", "namaste", "shalom", "salaam", "aloha", "hej", "olá",
            "привет", "γεια", "مرحبا", "안녕하세요", "こんにちは", "你好", "नमस्ते"
        ]
        
        found_greetings = []
        answer_lower = complete_answer.lower()
        for greeting in common_hellos:
            if greeting in answer_lower:
                found_greetings.append(greeting)
        
        print(f"🌍 Potential greetings found: {len(found_greetings)}")
        if found_greetings:
            print(f"   Examples: {', '.join(found_greetings[:5])}{'...' if len(found_greetings) > 5 else ''}")
        
        # Save results to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results = {
            "test_info": {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "model_mode": "deep (deepseek-reasoner preferred)",
                "service_class": "DeepSeekService"
            },
            "reasoning_content": {
                "full_text": complete_reasoning,
                "length": len(complete_reasoning),
                "chunks_received": len(reasoning_chunks)
            },
            "answer_content": {
                "full_text": complete_answer,
                "length": len(complete_answer),
                "chunks_received": len(answer_chunks)
            },
            "analysis": {
                "total_length": len(complete_reasoning + complete_answer),
                "greetings_found": found_greetings,
                "greeting_count": len(found_greetings)
            }
        }
        
        results_file = f"service_test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {results_file}")
        
        # Save reasoning and answer separately for easy reading
        if complete_reasoning:
            reasoning_file = f"service_reasoning_{timestamp}.txt"
            with open(reasoning_file, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n")
                f.write(complete_reasoning)
            print(f"💾 Reasoning saved to: {reasoning_file}")
        
        if complete_answer:
            answer_file = f"service_answer_{timestamp}.txt"
            with open(answer_file, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n")
                f.write(complete_answer)
            print(f"💾 Answer saved to: {answer_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Service test failed: {e}")
        return False

async def test_non_streaming_comparison():
    """
    Test the same query using non-streaming method for comparison.
    """
    query = "How many languages do you say 'hello'? Please show me 'hello' in each language you can."
    
    print_header("NON-STREAMING COMPARISON")
    
    # Initialize service
    service = DeepSeekService([], [])
    
    try:
        print("🔄 Testing non-streaming method...")
        
        # Test with 'deep' mode (prefers deepseek-reasoner)
        response = await service.async_chat_completion(
            query=query,
            system_message="You are a helpful assistant with knowledge of many languages.",
            search_mode="deep"
        )
        
        content = response.get('content', 'No content received')
        model_used = response.get('model', 'Unknown model')
        
        print(f"🤖 Model used: {model_used}")
        print_section("📄 NON-STREAMING RESPONSE", content, "🟡 ")
        print(f"📊 Response length: {len(content)} characters")
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"non_streaming_comparison_{timestamp}.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Model: {model_used}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n")
            f.write(content)
        
        print(f"💾 Comparison saved to: {comparison_file}")
        
    except Exception as e:
        print(f"❌ Non-streaming test failed: {e}")

async def main():
    """Main function to run all tests."""
    print("🌟 DEEPSEEK SERVICE REASONING & ANSWER TEST")
    print("=" * 60)
    print("This test uses the DeepSeekService class to demonstrate")
    print("the difference between reasoning and answer content.")
    print("=" * 60)
    
    # Run streaming test
    success = await test_streaming_with_reasoning()
    
    if success:
        print("\n🎉 Streaming test completed successfully!")
        
        # Run non-streaming comparison
        await test_non_streaming_comparison()
        
        print_header("TEST SUMMARY")
        print("✅ Successfully tested DeepSeek service reasoning functionality")
        print("📁 Check generated files for detailed analysis")
        print("🔍 Compare streaming vs non-streaming responses")
        print("🧠 Review reasoning process vs final answer")
        
    else:
        print("\n❌ Test failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
