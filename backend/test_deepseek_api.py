#!/usr/bin/env python3
"""
Test script for DeepSeek API connectivity.
This script tests both synchronous and asynchronous calls to the DeepSeek API.

Usage:
    python test_deepseek_api.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: The base URL for the DeepSeek API (default: https://api.deepseek.com)
"""

import os
import sys
import json
import asyncio
import argparse
import requests
import logging
from dotenv import load_dotenv
from openai import OpenAI
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='backend.log',
    filemode='a'  # Append mode
)
# Add console handler to see logs in the console as well
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)
logger.info("Test DeepSeek API script started")

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')

# Global variable for model override
model_override = "deepseek-chat"

def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "=" * 80 + "\n")

def test_environment():
    """Test if the environment variables are set correctly."""
    print("Testing environment variables...")
    logger.info("Testing environment variables")
    
    if not API_KEY:
        msg = "DEEPSEEK_API_KEY is not set. Please set it in your .env file or as an environment variable."
        print(f"? {msg}")
        logger.error(msg)
        return False
    else:
        msg = f"DEEPSEEK_API_KEY is set: {API_KEY[:5]}...{API_KEY[-5:] if len(API_KEY) > 10 else ''}"
        print(f"? {msg}")
        logger.info(msg)
    
    if not API_URL:
        msg = "DEEPSEEK_API_URL is not set. Using default: https://api.deepseek.com"
        print(f"? {msg}")
        logger.warning(msg)
    else:
        msg = f"DEEPSEEK_API_URL is set: {API_URL}"
        print(f"? {msg}")
        logger.info(msg)
    
    return True

def test_synchronous_request(query="Hello"):
    """
    Test a synchronous request to the DeepSeek API using the OpenAI SDK.
    
    Args:
        query (str): The query to send to the API
    """
    print_separator()
    print(f"Testing synchronous request with query: '{query}'")
    print(f"Using model: {model_override}")
    logger.info(f"Testing synchronous request with query: '{query}'")
    logger.info(f"Using model: {model_override}")
    
    try:
        print(f"Initializing OpenAI client with base URL: {API_URL}")
        logger.info(f"Initializing OpenAI client with base URL: {API_URL}")
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("Sending chat completion request...")
        logger.info("Sending chat completion request to DeepSeek API")
        response = client.chat.completions.create(
            model=model_override,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
            stream=False,
            timeout=30.0  # Add timeout parameter
        )
        
        print("? Synchronous request successful!")
        logger.info("Synchronous request to DeepSeek API successful")
        
        # Log the response details
        logger.info(f"DeepSeek API Response - Model: {response.model}")
        logger.info(f"DeepSeek API Response - ID: {response.id}")
        logger.info(f"DeepSeek API Response - Created: {response.created}")
        logger.info(f"DeepSeek API Response - Content: {response.choices[0].message.content}")
        
        print("\nResponse:")
        print(f"Model: {response.model}")
        print(f"ID: {response.id}")
        print(f"Created: {response.created}")
        print(f"Content: {response.choices[0].message.content}")
        
    except Exception as e:
        error_msg = f"Synchronous request failed with error: {e}"
        print(f"? {error_msg}")
        logger.error(error_msg)
        
        # Provide more detailed error information
        if "500" in str(e) and "Internal Server Error" in str(e):
            print("? This appears to be a server-side issue with the DeepSeek API.")
            print("? Suggestions:")
            print("  1. Try a different model with --model parameter (e.g., deepseek-chat, deepseek-reasoner)")
            print("  2. Check if your API key is valid and has access to the requested model")
            print("  3. The DeepSeek API service might be experiencing issues - try again later")
            logger.error("500 Internal Server Error detected - likely a server-side issue")
        elif "timeout" in str(e).lower():
            print("? Request timed out. The DeepSeek API might be experiencing high load.")
            print("? Suggestions:")
            print("  1. Try again later when the service might be less busy")
            print("  2. Check your network connection")
            logger.error("Timeout error detected")
        elif "unauthorized" in str(e).lower() or "authentication" in str(e).lower():
            print("? Authentication error. Your API key might be invalid or expired.")
            print("? Suggestions:")
            print("  1. Check your DEEPSEEK_API_KEY environment variable")
            print("  2. Obtain a new API key if necessary")
            logger.error("Authentication error detected")

async def test_asynchronous_request(query="Hello"):
    """
    Test an asynchronous request to the DeepSeek API using the AsyncOpenAI client.
    
    Args:
        query (str): The query to send to the API
    """
    print_separator()
    print(f"Testing asynchronous request with query: '{query}'")
    print(f"Using model: {model_override}")
    logger.info(f"Testing asynchronous request with query: '{query}'")
    logger.info(f"Using model: {model_override}")
    
    try:
        print(f"Initializing AsyncOpenAI client with base URL: {API_URL}")
        logger.info(f"Initializing AsyncOpenAI client with base URL: {API_URL}")
        async_client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("Sending asynchronous chat completion request...")
        logger.info("Sending asynchronous chat completion request to DeepSeek API")
        response = await async_client.chat.completions.create(
            model=model_override,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
            stream=False,
            timeout=30.0  # Add timeout parameter
        )
        
        print("? Asynchronous request successful!")
        logger.info("Asynchronous request to DeepSeek API successful")
        
        # Log the response details
        logger.info(f"DeepSeek API Response (Async) - Model: {response.model}")
        logger.info(f"DeepSeek API Response (Async) - ID: {response.id}")
        logger.info(f"DeepSeek API Response (Async) - Created: {response.created}")
        logger.info(f"DeepSeek API Response (Async) - Content: {response.choices[0].message.content}")
        
        print("\nResponse:")
        print(f"Model: {response.model}")
        print(f"ID: {response.id}")
        print(f"Created: {response.created}")
        print(f"Content: {response.choices[0].message.content}")
        
    except Exception as e:
        error_msg = f"Asynchronous request failed with error: {e}"
        print(f"? {error_msg}")
        logger.error(error_msg)
        
        # Provide more detailed error information
        if "500" in str(e) and "Internal Server Error" in str(e):
            print("? This appears to be a server-side issue with the DeepSeek API.")
            print("? Suggestions:")
            print("  1. Try a different model with --model parameter (e.g., deepseek-chat, deepseek-reasoner)")
            print("  2. Check if your API key is valid and has access to the requested model")
            print("  3. The DeepSeek API service might be experiencing issues - try again later")
            logger.error("500 Internal Server Error detected - likely a server-side issue")
        elif "timeout" in str(e).lower():
            print("? Request timed out. The DeepSeek API might be experiencing high load.")
            print("? Suggestions:")
            print("  1. Try again later when the service might be less busy")
            print("  2. Check your network connection")
            logger.error("Timeout error detected")
        elif "unauthorized" in str(e).lower() or "authentication" in str(e).lower():
            print("? Authentication error. Your API key might be invalid or expired.")
            print("? Suggestions:")
            print("  1. Check your DEEPSEEK_API_KEY environment variable")
            print("  2. Obtain a new API key if necessary")
            logger.error("Authentication error detected")

def test_network_connectivity():
    """Test basic network connectivity to common websites."""
    print_separator()
    print("Testing network connectivity...")
    logger.info("Testing network connectivity")
    
    sites = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.microsoft.com"
    ]
    
    for site in sites:
        try:
            response = requests.get(site, timeout=5)
            if response.status_code == 200:
                msg = f"Successfully connected to {site}"
                print(f"? {msg}")
                logger.info(msg)
            else:
                msg = f"Failed to connect to {site} with status code {response.status_code}"
                print(f"? {msg}")
                logger.warning(msg)
        except requests.exceptions.RequestException as e:
            msg = f"Failed to connect to {site} with error: {e}"
            print(f"? {msg}")
            logger.error(msg)

async def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Test DeepSeek API connectivity')
    parser.add_argument('--query', type=str, default="Hello",
                        help='Query to send to the DeepSeek API')
    parser.add_argument('--model', type=str, default="deepseek-chat",
                        help='Model to use for API requests (e.g., deepseek-chat, deepseek-reasoner)')
    parser.add_argument('--skip-sync', action='store_true',
                        help='Skip synchronous API test')
    parser.add_argument('--skip-async', action='store_true',
                        help='Skip asynchronous API test')
    parser.add_argument('--skip-network', action='store_true',
                        help='Skip network connectivity test')
    args = parser.parse_args()
    
    print("\nDEEPSEEK API CONNECTIVITY TEST")
    print("=============================\n")
    logger.info("Starting DeepSeek API connectivity test")
    logger.info(f"Using model: {args.model}")
    
    # Test environment variables
    if not test_environment():
        msg = "Environment variables not set correctly. Exiting."
        print(f"\n? {msg}")
        logger.error(msg)
        sys.exit(1)
    
    # Test network connectivity if not skipped
    if not args.skip_network:
        test_network_connectivity()
    else:
        print("Skipping network connectivity test")
        logger.info("Skipping network connectivity test")
    
    # Test synchronous request if not skipped
    if not args.skip_sync:
        # Override the model in the test function
        global model_override
        model_override = args.model
        test_synchronous_request(args.query)
    else:
        print("Skipping synchronous API test")
        logger.info("Skipping synchronous API test")
    
    # Test asynchronous request if not skipped
    if not args.skip_async:
        await test_asynchronous_request(args.query)
    else:
        print("Skipping asynchronous API test")
        logger.info("Skipping asynchronous API test")
    
    print_separator()
    print("All tests completed!")
    logger.info("DeepSeek API connectivity test completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
