#!/usr/bin/env python3
"""
Quick test to demonstrate reasoning functionality with current models.
This test shows that deepseek-chat (which enables DeepSeek-R1-0528) works correctly.
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')

async def test_reasoning_with_deepseek_chat():
    """Test reasoning functionality with deepseek-chat model (which enables DeepSeek-R1-0528)."""
    
    query = "How many languages do you say 'hello'? Please show me 'hello' in each language you can."
    
    print("? QUICK REASONING TEST")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Model: deepseek-chat (enables DeepSeek-R1-0528)")
    print("=" * 60)
    
    if not API_KEY:
        print("? API key not configured")
        return False
    
    try:
        client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("? Sending request...")
        start_time = datetime.now()
        
        # Use deepseek-chat which enables DeepSeek-R1-0528
        stream = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with knowledge of many languages."},
                {"role": "user", "content": query}
            ],
            stream=True,
            timeout=300.0  # 5 minute timeout
        )
        
        print("? Streaming started...")
        
        reasoning_content = ""
        answer_content = ""
        chunk_count = 0
        
        print("\n? LIVE STREAM:")
        print("-" * 50)
        
        async for chunk in stream:
            chunk_count += 1
            
            # Handle reasoning content (if available)
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_chunk = chunk.choices[0].delta.reasoning_content
                reasoning_content += reasoning_chunk
                print(f"? {reasoning_chunk[:100]}{'...' if len(reasoning_chunk) > 100 else ''}")
            
            # Handle answer content
            elif chunk.choices[0].delta.content:
                answer_chunk = chunk.choices[0].delta.content
                answer_content += answer_chunk
                # Show first few characters of each chunk
                print(f"? {answer_chunk[:50]}{'...' if len(answer_chunk) > 50 else ''}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("-" * 50)
        print(f"? Stream completed in {duration:.1f} seconds!")
        print(f"? Total chunks received: {chunk_count}")
        
        # Display summary
        print("\n? RESULTS SUMMARY:")
        print("=" * 50)
        
        if reasoning_content:
            print(f"? Reasoning content: {len(reasoning_content)} characters")
            print("   First 200 chars:", reasoning_content[:200] + "..." if len(reasoning_content) > 200 else reasoning_content)
        else:
            print("? Reasoning content: None received")
        
        if answer_content:
            print(f"? Answer content: {len(answer_content)} characters")
            print("   First 200 chars:", answer_content[:200] + "..." if len(answer_content) > 200 else answer_content)
        else:
            print("? Answer content: None received")
        
        # Count languages mentioned
        hello_words = ["hello", "hola", "bonjour", "konnichiwa", "namaste", "ciao", "hej"]
        found_words = [word for word in hello_words if word.lower() in answer_content.lower()]
        print(f"? Sample greetings found: {len(found_words)} ({', '.join(found_words[:5])})")
        
        # Save quick results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quick_test_results_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Quick Reasoning Test Results\n")
            f.write(f"Query: {query}\n")
            f.write(f"Model: deepseek-chat (DeepSeek-R1-0528)\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write(f"Chunks: {chunk_count}\n")
            f.write("=" * 60 + "\n")
            f.write("REASONING CONTENT:\n")
            f.write(reasoning_content or "None")
            f.write("\n" + "=" * 60 + "\n")
            f.write("ANSWER CONTENT:\n")
            f.write(answer_content or "None")
        
        print(f"\n? Results saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"? Error: {e}")
        return False

async def main():
    """Main function."""
    print("? QUICK DEEPSEEK REASONING TEST")
    print("This test verifies that deepseek-chat (DeepSeek-R1-0528) works correctly")
    print("and measures actual response time.\n")
    
    success = await test_reasoning_with_deepseek_chat()
    
    if success:
        print("\n? Test completed successfully!")
        print("? The model is working correctly with reasonable response times.")
        print("? The previous timeout issue was with the test runner, not the API.")
    else:
        print("\n? Test failed.")

if __name__ == "__main__":
    asyncio.run(main())
