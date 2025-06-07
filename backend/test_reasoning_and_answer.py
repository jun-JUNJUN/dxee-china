#!/usr/bin/env python3
"""
Test script to demonstrate both reasoning content and answer content from DeepSeek API.
This script specifically tests the query about saying 'hello' in different languages
to showcase how the deepseek-reasoner model provides both reasoning and final answer.

Usage:
    python test_reasoning_and_answer.py

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
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reasoning_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API credentials
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')

def print_separator(title=""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("\n" + "=" * 60)

def print_content_box(title, content, color_code=""):
    """Print content in a formatted box."""
    print(f"\n{color_code}┌─ {title} " + "─" * (50 - len(title)) + "┐")
    
    # Split content into lines and wrap long lines
    lines = content.split('\n')
    for line in lines:
        if len(line) <= 48:
            print(f"│ {line:<48} │")
        else:
            # Wrap long lines
            words = line.split(' ')
            current_line = ""
            for word in words:
                if len(current_line + word) <= 48:
                    current_line += word + " "
                else:
                    if current_line:
                        print(f"│ {current_line.strip():<48} │")
                    current_line = word + " "
            if current_line:
                print(f"│ {current_line.strip():<48} │")
    
    print(f"└{'─' * 50}┘\033[0m")

async def test_reasoning_and_answer():
    """
    Test the DeepSeek API with a specific query to demonstrate both reasoning and answer content.
    """
    # The specific query requested by the user
    query = "How many languages do you say 'hello'? Please show me 'hello' in each language you can."
    
    print_separator("DEEPSEEK REASONING & ANSWER TEST")
    print(f"Query: {query}")
    print(f"Model: deepseek-reasoner (for reasoning content)")
    print(f"API URL: {API_URL}")
    
    if not API_KEY:
        print("❌ DEEPSEEK_API_KEY is not set!")
        return False
    
    try:
        # Initialize async client
        client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("\n🚀 Sending request to DeepSeek API...")
        logger.info(f"Sending request with query: {query}")
        
        # Use deepseek-reasoner model to get both reasoning and answer
        stream = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant with knowledge of many languages. When asked about languages, provide comprehensive and accurate information."
                },
                {"role": "user", "content": query}
            ],
            stream=True,
            timeout=60.0
        )
        
        print("✅ Request successful! Streaming response...")
        
        # Collect reasoning and answer content separately
        reasoning_content = ""
        answer_content = ""
        
        print_separator("LIVE STREAMING RESPONSE")
        
        async for chunk in stream:
            # Handle reasoning content (thinking process)
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_chunk = chunk.choices[0].delta.reasoning_content
                reasoning_content += reasoning_chunk
                print(f"🧠 Reasoning: {reasoning_chunk}", end="", flush=True)
            
            # Handle regular content (final answer)
            elif chunk.choices[0].delta.content:
                answer_chunk = chunk.choices[0].delta.content
                answer_content += answer_chunk
                print(f"💬 Answer: {answer_chunk}", end="", flush=True)
        
        print("\n")
        print_separator("COMPLETE RESPONSE ANALYSIS")
        
        # Display reasoning content
        if reasoning_content:
            print_content_box("🧠 REASONING CONTENT (Thinking Process)", reasoning_content, "\033[94m")  # Blue
            logger.info(f"Reasoning content length: {len(reasoning_content)} characters")
        else:
            print("⚠️  No reasoning content received (model may not support it)")
        
        # Display answer content
        if answer_content:
            print_content_box("💬 ANSWER CONTENT (Final Response)", answer_content, "\033[92m")  # Green
            logger.info(f"Answer content length: {len(answer_content)} characters")
        else:
            print("❌ No answer content received")
        
        # Analysis and statistics
        print_separator("RESPONSE STATISTICS")
        
        # Count languages mentioned in the answer
        hello_variations = [
            "hello", "hola", "bonjour", "guten tag", "ciao", "konnichiwa", 
            "ni hao", "namaste", "shalom", "salaam", "aloha", "hej", 
            "olá", "привет", "γεια", "مرحبا", "안녕하세요", "こんにちは",
            "你好", "नमस्ते", "سلام", "שלום"
        ]
        
        found_languages = []
        answer_lower = answer_content.lower()
        for variation in hello_variations:
            if variation in answer_lower:
                found_languages.append(variation)
        
        print(f"📊 Total response length: {len(reasoning_content + answer_content)} characters")
        print(f"🧠 Reasoning content: {len(reasoning_content)} characters")
        print(f"💬 Answer content: {len(answer_content)} characters")
        print(f"🌍 Languages detected in answer: {len(found_languages)}")
        if found_languages:
            print(f"   Found: {', '.join(found_languages[:10])}{'...' if len(found_languages) > 10 else ''}")
        
        # Save to files for detailed analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if reasoning_content:
            reasoning_file = f"reasoning_content_{timestamp}.txt"
            with open(reasoning_file, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: deepseek-reasoner\n")
                f.write("=" * 50 + "\n")
                f.write(reasoning_content)
            print(f"💾 Reasoning content saved to: {reasoning_file}")
        
        if answer_content:
            answer_file = f"answer_content_{timestamp}.txt"
            with open(answer_file, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: deepseek-reasoner\n")
                f.write("=" * 50 + "\n")
                f.write(answer_content)
            print(f"💾 Answer content saved to: {answer_file}")
        
        # Create combined JSON output
        combined_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "model": "deepseek-reasoner",
            "reasoning_content": reasoning_content,
            "answer_content": answer_content,
            "statistics": {
                "total_length": len(reasoning_content + answer_content),
                "reasoning_length": len(reasoning_content),
                "answer_length": len(answer_content),
                "languages_detected": len(found_languages),
                "found_languages": found_languages
            }
        }
        
        json_file = f"reasoning_and_answer_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Combined data saved to: {json_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Test failed with error: {e}")
        
        # Provide specific error guidance
        if "500" in str(e):
            print("💡 Suggestion: Try again later, the API might be experiencing issues")
        elif "unauthorized" in str(e).lower():
            print("💡 Suggestion: Check your DEEPSEEK_API_KEY environment variable")
        elif "timeout" in str(e).lower():
            print("💡 Suggestion: The request timed out, try again with a simpler query")
        
        return False

async def test_comparison_with_regular_model():
    """
    Test the same query with deepseek-chat model for comparison.
    """
    query = "How many languages do you say 'hello'? Please show me 'hello' in each language you can."
    
    print_separator("COMPARISON WITH REGULAR MODEL")
    print("Testing the same query with deepseek-chat (no reasoning)")
    
    try:
        client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
        
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant with knowledge of many languages."
                },
                {"role": "user", "content": query}
            ],
            stream=False,
            timeout=30.0
        )
        
        regular_content = response.choices[0].message.content
        print_content_box("🤖 DEEPSEEK-CHAT RESPONSE (No Reasoning)", regular_content, "\033[93m")  # Yellow
        
        print(f"📊 Regular model response length: {len(regular_content)} characters")
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"model_comparison_{timestamp}.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n")
            f.write("DEEPSEEK-CHAT RESPONSE:\n")
            f.write(regular_content)
        
        print(f"💾 Comparison saved to: {comparison_file}")
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")

async def main():
    """Main function to run the reasoning and answer test."""
    print("🌟 DEEPSEEK REASONING & ANSWER DEMONSTRATION")
    print("=" * 60)
    print("This test demonstrates the difference between reasoning content")
    print("and answer content using the deepseek-reasoner model.")
    print("=" * 60)
    
    # Test environment
    if not API_KEY:
        print("❌ Please set DEEPSEEK_API_KEY in your .env file")
        sys.exit(1)
    
    print(f"✅ API Key configured: {API_KEY[:8]}...{API_KEY[-4:]}")
    
    # Run the main test
    success = await test_reasoning_and_answer()
    
    if success:
        print("\n🎉 Test completed successfully!")
        
        # Run comparison test
        try:
            await test_comparison_with_regular_model()
        except Exception as e:
            print(f"⚠️  Comparison test failed: {e}")
        
        print_separator("TEST SUMMARY")
        print("✅ Successfully demonstrated reasoning and answer content")
        print("📁 Check the generated files for detailed analysis")
        print("🔍 Compare the reasoning process vs final answer")
        print("📊 Review the statistics and language detection results")
        
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
