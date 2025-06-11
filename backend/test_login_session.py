#!/usr/bin/env python3
"""
Test script to simulate login and check session
"""
import asyncio
import os
import requests
from dotenv import load_dotenv

async def test_login_session():
    """Test login session functionality"""
    print("? Testing Login Session...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    base_url = "http://localhost:8100"
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    print("1. Testing session check without login...")
    response = session.get(f"{base_url}/auth/session")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n2. Testing Google OAuth POST (simulating successful Google login)...")
    
    # Simulate a Google ID token (this would normally come from Google)
    # For testing, we'll use a mock payload
    mock_google_data = {
        "id_token": "mock_token_for_testing"
    }
    
    try:
        response = session.post(f"{base_url}/auth/google", json=mock_google_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Testing session check after login attempt...")
    response = session.get(f"{base_url}/auth/session")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n4. Checking cookies...")
    for cookie in session.cookies:
        print(f"   Cookie: {cookie.name} = {cookie.value[:20]}...")

if __name__ == "__main__":
    asyncio.run(test_login_session())
