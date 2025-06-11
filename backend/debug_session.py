#!/usr/bin/env python3
"""
Debug script to check session cookies and user validation
"""
import asyncio
import os
import requests
from dotenv import load_dotenv
import tornado.web
from app.service.mongodb_service import MongoDBService
import tornado.ioloop

async def debug_session():
    """Debug session functionality"""
    print("? Debugging Session...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    base_url = "http://localhost:8100"
    known_user_id = "6847ebd29dc6c51d64edbcde"
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    print("1. Testing session check...")
    response = session.get(f"{base_url}/auth/session")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n2. Checking cookies from browser...")
    print("   Current cookies:")
    for cookie in session.cookies:
        print(f"   - {cookie.name} = {cookie.value}")
    
    print("\n3. Testing MongoDB connection and user lookup...")
    try:
        # Initialize MongoDB service
        io_loop = tornado.ioloop.IOLoop.current()
        mongodb = MongoDBService(io_loop=io_loop)
        
        user = await mongodb.get_user_by_id(known_user_id)
        
        if user:
            print(f"   ? User found in database:")
            print(f"      ID: {user['_id']}")
            print(f"      Email: {user['email']}")
            print(f"      Username: {user['username']}")
        else:
            print(f"   ? User not found in database")
            
    except Exception as e:
        print(f"   ? Error connecting to MongoDB: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Testing cookie secret...")
    cookie_secret = os.environ.get('AUTH_SECRET_KEY', 'default_secret_key_change_in_production')
    print(f"   Current cookie secret: {cookie_secret[:10]}...")
    
    print("\n5. Manual session test with known user ID...")
    # Simulate setting a secure cookie manually
    try:
        import base64
        import hmac
        import hashlib
        import time
        
        # Create a mock secure cookie value
        user_id = known_user_id
        timestamp = str(int(time.time()))
        value = base64.b64encode(user_id.encode()).decode()
        signature = base64.b64encode(
            hmac.new(
                cookie_secret.encode(),
                f"{value}|{timestamp}".encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        
        secure_cookie_value = f"{value}|{timestamp}|{signature}"
        print(f"   Generated secure cookie: {secure_cookie_value[:50]}...")
        
        # Set the cookie and test
        session.cookies.set('user_id', secure_cookie_value)
        
        print("\n6. Testing session check with manual cookie...")
        response = session.get(f"{base_url}/auth/session")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
    except Exception as e:
        print(f"   ? Error creating manual cookie: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_session())
