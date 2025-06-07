#!/usr/bin/env python3
"""
Simple test to demonstrate reasoning content with a shorter query and longer timeout.
This test uses a simpler query to show reasoning vs answer content more quickly.
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

async def test_simple_reasoning():
    """Test with a simpler query that should respond faster."""
    
    # Use a simpler query that still triggers reasoning
    query = "What are 5 ways to say hello in different languages?"
    
    print("? SIMPLE REASONING TEST")
    print("=" * 50)
    print(f"Query: {query}")
    print(f"Model: deepseek-reasoner")
    print("=" * 50)
    
    if not API_KEY:
        print("? API key not configured")
        return
    
    try:
        client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("? Sending request...")
        
        # Use longer timeout for reasoning model
        stream = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            stream=True,
            timeout=180.0  # 3 minute timeout
        )
        
        print("? Streaming started...")
        
        reasoning_content = ""
        answer_content = ""
        
        print("\n? LIVE STREAM:")
        print("-" * 30)
        
        async for chunk in stream:
            # Handle reasoning content
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_chunk = chunk.choices[0].delta.reasoning_content
                reasoning_content += reasoning_chunk
                print(f"? {reasoning_chunk}", end="", flush=True)
            
            # Handle answer content
            elif chunk.choices[0].delta.content:
                answer_chunk = chunk.choices[0].delta.content
                answer_content += answer_chunk
                print(f"? {answer_chunk}", end="", flush=True)
        
        print("\n" + "-" * 30)
        print("? Stream completed!")
        
        # Display results
        print("\n? REASONING CONTENT:")
        print("=" * 40)
        if reasoning_content:
            print(reasoning_content)
            print(f"\nLength: {len(reasoning_content)} characters")
        else:
            print("No reasoning content received")
        
        print("\n? ANSWER CONTENT:")
        print("=" * 40)
        if answer_content:
            print(answer_content)
            print(f"\nLength: {len(answer_content)} characters")
        else:
            print("No answer content received")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if reasoning_content or answer_content:
            filename = f"simple_reasoning_test_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
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
    print("? SIMPLE DEEPSEEK REASONING TEST")
    print("This test uses a simpler query to demonstrate reasoning vs answer content")
    print("with a longer timeout to accommodate the reasoning model.\n")
    
    success = await test_simple_reasoning()
    
    if success:
        print("\n? Test completed successfully!")
        print("? Check the generated file for complete reasoning and answer content.")
    else:
        print("\n? Test failed.")

if __name__ == "__main__":
    asyncio.run(main())
