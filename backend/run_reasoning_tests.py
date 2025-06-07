#!/usr/bin/env python3
"""
Simple script to run the reasoning and answer content tests.
This script provides an easy way to execute both test files and compare results.

Usage:
    python run_reasoning_tests.py [--test-type TYPE]

Options:
    --test-type: Choose 'direct', 'service', or 'both' (default: both)
"""

import os
import sys
import asyncio
import argparse
import subprocess
from datetime import datetime

def print_banner():
    """Print a welcome banner."""
    print("🌟" * 30)
    print("🧠 DEEPSEEK REASONING & ANSWER CONTENT TESTS 🧠")
    print("🌟" * 30)
    print()
    print("This script demonstrates how DeepSeek's reasoning model")
    print("provides both thinking process (reasoning) and final answer.")
    print()
    print("Query: 'How many languages do you say 'hello'?")
    print("       Please show me 'hello' in each language you can.'")
    print()

def check_environment():
    """Check if the environment is properly configured."""
    print("🔍 Checking environment...")
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Please create one with DEEPSEEK_API_KEY")
        return False
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment")
        print("   Please add DEEPSEEK_API_KEY=your_key_here to your .env file")
        return False
    
    print(f"✅ API Key configured: {api_key[:8]}...{api_key[-4:]}")
    return True

def run_test_script(script_name, description):
    """Run a test script and capture its output."""
    print(f"\n{'=' * 20} {description} {'=' * 20}")
    print(f"🚀 Running: {script_name}")
    
    try:
        # Run the script with longer timeout for reasoning models
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for reasoning models
        )
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            print("\n📄 Output:")
            print(result.stdout)
            if result.stderr:
                print("\n⚠️  Warnings/Info:")
                print(result.stderr)
            return True
        else:
            print(f"❌ Test failed with return code: {result.returncode}")
            print("\n📄 Output:")
            print(result.stdout)
            if result.stderr:
                print("\n❌ Errors:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

async def run_service_test_directly():
    """Run the service test directly (async)."""
    print(f"\n{'=' * 20} DIRECT SERVICE TEST {'=' * 20}")
    print("🚀 Running service test directly...")
    
    try:
        # Import and run the service test
        sys.path.append('.')
        from test_service_reasoning import main as service_main
        
        await service_main()
        print("✅ Direct service test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Direct service test failed: {e}")
        return False

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description='Run DeepSeek reasoning tests')
    parser.add_argument('--test-type', choices=['direct', 'service', 'both'], 
                       default='both', help='Type of test to run')
    args = parser.parse_args()
    
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n🎯 Test Configuration:")
    print(f"   Test Type: {args.test_type}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run tests based on selection
    if args.test_type in ['direct', 'both']:
        print("\n" + "🔵" * 50)
        print("DIRECT API TEST - Uses OpenAI client directly")
        print("🔵" * 50)
        success = run_test_script('test_reasoning_and_answer.py', 'Direct API Test')
        results.append(('Direct API Test', success))
    
    if args.test_type in ['service', 'both']:
        print("\n" + "🟢" * 50)
        print("SERVICE CLASS TEST - Uses DeepSeekService class")
        print("🟢" * 50)
        success = run_test_script('test_service_reasoning.py', 'Service Class Test')
        results.append(('Service Class Test', success))
    
    # Summary
    print("\n" + "🌟" * 50)
    print("TEST SUMMARY")
    print("🌟" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n📁 Check the generated files for detailed analysis:")
        print("   - reasoning_content_*.txt - Thinking process")
        print("   - answer_content_*.txt - Final answers")
        print("   - *.json - Complete structured data")
        print("   - *.log - Detailed logs")
        
        print("\n🔍 Key things to observe:")
        print("   1. Reasoning shows the AI's thinking process")
        print("   2. Answer contains the final formatted response")
        print("   3. Different models may provide different levels of reasoning")
        print("   4. Streaming captures both types of content separately")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
