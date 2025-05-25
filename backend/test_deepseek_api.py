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
from dotenv import load_dotenv
from openai import OpenAI
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')

def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "=" * 80 + "\n")

def test_environment():
    """Test if the environment variables are set correctly."""
    print("Testing environment variables...")
    
    if not API_KEY:
        print("? DEEPSEEK_API_KEY is not set. Please set it in your .env file or as an environment variable.")
        return False
    else:
        print(f"? DEEPSEEK_API_KEY is set: {API_KEY[:5]}...{API_KEY[-5:] if len(API_KEY) > 10 else ''}")
    
    if not API_URL:
        print("? DEEPSEEK_API_URL is not set. Using default: https://api.deepseek.com")
    else:
        print(f"? DEEPSEEK_API_URL is set: {API_URL}")
    
    return True

def test_synchronous_request(query="Hello"):
    """
    Test a synchronous request to the DeepSeek API using the OpenAI SDK.
    
    Args:
        query (str): The query to send to the API
    """
    print_separator()
    print(f"Testing synchronous request with query: '{query}'")
    
    try:
        print(f"Initializing OpenAI client with base URL: {API_URL}")
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("Sending chat completion request...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
            stream=False
        )
        
        print("? Synchronous request successful!")
        print("\nResponse:")
        print(f"Model: {response.model}")
        print(f"ID: {response.id}")
        print(f"Created: {response.created}")
        print(f"Content: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"? Synchronous request failed with error: {e}")

async def test_asynchronous_request(query="Hello"):
    """
    Test an asynchronous request to the DeepSeek API using the AsyncOpenAI client.
    
    Args:
        query (str): The query to send to the API
    """
    print_separator()
    print(f"Testing asynchronous request with query: '{query}'")
    
    try:
        print(f"Initializing AsyncOpenAI client with base URL: {API_URL}")
        async_client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)
        
        print("Sending asynchronous chat completion request...")
        response = await async_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
            stream=False
        )
        
        print("? Asynchronous request successful!")
        print("\nResponse:")
        print(f"Model: {response.model}")
        print(f"ID: {response.id}")
        print(f"Created: {response.created}")
        print(f"Content: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"? Asynchronous request failed with error: {e}")

def test_network_connectivity():
    """Test basic network connectivity to common websites."""
    print_separator()
    print("Testing network connectivity...")
    
    sites = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.microsoft.com"
    ]
    
    for site in sites:
        try:
            response = requests.get(site, timeout=5)
            if response.status_code == 200:
                print(f"? Successfully connected to {site}")
            else:
                print(f"? Failed to connect to {site} with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"? Failed to connect to {site} with error: {e}")

async def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Test DeepSeek API connectivity')
    parser.add_argument('--query', type=str, default="Hello", 
                        help='Query to send to the DeepSeek API')
    args = parser.parse_args()
    
    print("\nDEEPSEEK API CONNECTIVITY TEST")
    print("=============================\n")
    
    # Test environment variables
    if not test_environment():
        print("\n? Environment variables not set correctly. Exiting.")
        sys.exit(1)
    
    # Test network connectivity
    test_network_connectivity()
    
    # Test synchronous request
    test_synchronous_request(args.query)
    
    # Test asynchronous request
    await test_asynchronous_request(args.query)
    
    print_separator()
    print("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
