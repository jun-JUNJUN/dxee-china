#!/usr/bin/env python3
"""
Debug script to test the session check handler directly
"""
import asyncio
import os
import requests
import json
from dotenv import load_dotenv

async def debug_session_handler():
    """Debug session handler"""
    print("🔍 Debugging Session Handler...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    base_url = "http://localhost:8100"
    
    print("1. First, let's try to login with Google OAuth to get a real session...")
    
    # Try to get a real login session first
    session = requests.Session()
    
    # Check if we can access the main page
    print("2. Accessing main page to see if server is running...")
    try:
        response = session.get(base_url)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Server is running")
        else:
            print("   ❌ Server might not be running")
            return
    except Exception as e:
        print(f"   ❌ Error accessing server: {e}")
        return
    
    print("\n3. Testing session check endpoint...")
    response = session.get(f"{base_url}/auth/session")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print(f"   Headers: {dict(response.headers)}")
    
    print("\n4. Checking if there are any existing cookies...")
    for cookie in session.cookies:
        print(f"   Cookie: {cookie.name} = {cookie.value[:20]}...")
    
    print("\n5. Let's try to manually create a login session...")
    print("   (This would require a real Google OAuth flow)")
    
    # Let's check what the Google OAuth endpoint expects
    print("\n6. Checking Google OAuth endpoint...")
    try:
        response = session.get(f"{base_url}/auth/google")
        print(f"   Status: {response.status_code}")
        if response.status_code == 302:
            print(f"   Redirect to: {response.headers.get('Location', 'Unknown')}")
        else:
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_session_handler())
