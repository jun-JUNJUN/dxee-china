#!/usr/bin/env python3
"""
Live streaming test to show reasoning content in real-time in the terminal.
This test provides a better visual experience for watching reasoning content stream.
"""

import os
import asyncio
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Configure minimal logging to avoid cluttering the stream
logging.basicConfig(level=logging.WARNING)

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')

def print_header():
    """Print a nice header for the streaming test."""
    print("\n" + "?" * 60)
    print("?" + " " * 20 + "LIVE REASONING STREAM" + " " * 20 + "?")
    print("?" * 60)
    print()

def print_separator(title, color=""):
    """Print a separator with title."""
    print(f"\n{color}{'=' * 20} {title} {'=' * 20}\033[0m")

async def test_live_reasoning_stream():
    """Test live streaming of reasoning content with real-time display."""
    
    query = "How many languages do you say 'hello'? Please show me 'hello' in each language you can."
    
    print_header()
    print(f"? Query: {query}")
    print(f"? Model: deepseek-reasoner (for reasoning content)")
    print(f"? API: {API_URL}")
    
    if not API_KEY:
        print("? DEEPSEEK_API_KEY not configured!")
        return False
    
    print(f"? API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    
    try:
        client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("\n? Initiating streaming request...")
        start_time = datetime.now()
        
        # Use deepseek-reasoner for reasoning content
        stream = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant with extensive knowledge of world languages and cultures."
                },
                {"role": "user", "content": query}
            ],
            stream=True,
            timeout=300.0  # 5 minute timeout
        )
        
        print("? Stream established! Watching for content...\n")
        
        # Counters and accumulators
        reasoning_chunks = 0
        answer_chunks = 0
        reasoning_content = ""
        answer_content = ""
        
        # Visual indicators
        reasoning_active = False
        answer_active = False
        
        print_separator("? REASONING CONTENT (AI's Thinking Process)", "\033[94m")  # Blue
        
        async for chunk in stream:
            # Handle reasoning content
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_chunk = chunk.choices[0].delta.reasoning_content
                reasoning_content += reasoning_chunk
                reasoning_chunks += 1
                
                if not reasoning_active:
                    reasoning_active = True
                
                # Print reasoning content in blue with prefix
                print(f"\033[94m? {reasoning_chunk}\033[0m", end="", flush=True)
            
            # Handle answer content
            elif chunk.choices[0].delta.content:
                answer_chunk = chunk.choices[0].delta.content
                answer_content += answer_chunk
                answer_chunks += 1
                
                if not answer_active:
                    answer_active = True
                    print_separator("? ANSWER CONTENT (Final Response)", "\033[92m")  # Green
                
                # Print answer content in green with prefix
                print(f"\033[92m? {answer_chunk}\033[0m", end="", flush=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        print("\n")
        print_separator("? STREAMING SUMMARY", "\033[93m")  # Yellow
        
        print(f"??  Total duration: {duration:.1f} seconds")
        print(f"? Reasoning chunks: {reasoning_chunks}")
        print(f"? Answer chunks: {answer_chunks}")
        print(f"? Reasoning length: {len(reasoning_content)} characters")
        print(f"? Answer length: {len(answer_content)} characters")
        print(f"? Total content: {len(reasoning_content + answer_content)} characters")
        
        # Language analysis
        hello_words = ["hello", "hola", "bonjour", "konnichiwa", "namaste", "ciao", "shalom", "salaam"]
        found_greetings = [word for word in hello_words if word.lower() in answer_content.lower()]
        print(f"? Sample greetings found: {len(found_greetings)} ({', '.join(found_greetings[:5])})")
        
        # Save timestamped results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save reasoning content
        if reasoning_content:
            reasoning_file = f"live_reasoning_{timestamp}.txt"
            with open(reasoning_file, 'w', encoding='utf-8') as f:
                f.write(f"Live Reasoning Stream Test\n")
                f.write(f"Query: {query}\n")
                f.write(f"Duration: {duration:.1f} seconds\n")
                f.write(f"Chunks: {reasoning_chunks}\n")
                f.write("=" * 60 + "\n")
                f.write(reasoning_content)
            print(f"? Reasoning saved: {reasoning_file}")
        
        # Save answer content
        if answer_content:
            answer_file = f"live_answer_{timestamp}.txt"
            with open(answer_file, 'w', encoding='utf-8') as f:
                f.write(f"Live Answer Stream Test\n")
                f.write(f"Query: {query}\n")
                f.write(f"Duration: {duration:.1f} seconds\n")
                f.write(f"Chunks: {answer_chunks}\n")
                f.write("=" * 60 + "\n")
                f.write(answer_content)
            print(f"? Answer saved: {answer_file}")
        
        print("\n? Live streaming test completed successfully!")
        print("? You can see how the AI's reasoning process unfolds in real-time")
        print("? Notice the difference between thinking (reasoning) and final response (answer)")
        
        return True
        
    except Exception as e:
        print(f"\n? Error during streaming: {e}")
        return False

async def main():
    """Main function."""
    print("? DEEPSEEK LIVE REASONING STREAM TEST")
    print("This test shows reasoning content streaming in real-time in your terminal.")
    print("You'll see the AI's thinking process as it happens, followed by the final answer.")
    
    success = await test_live_reasoning_stream()
    
    if not success:
        print("\n? Test failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n??  Test interrupted by user (Ctrl+C)")
        print("? Any partial content may have been saved to files")
