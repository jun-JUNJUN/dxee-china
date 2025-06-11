#!/usr/bin/env python3
"""
Script to help diagnose and fix session cookie issues
"""
import asyncio
import os
import requests
import json
from dotenv import load_dotenv

async def fix_session_issue():
    """Help fix session cookie issues"""
    print("? Session Cookie Issue Diagnosis & Fix")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    base_url = "http://localhost:8100"
    
    print("ISSUE DIAGNOSIS:")
    print("You mentioned cookies are present but session check returns 'NOT LOGGED IN'")
    print("Our debug shows: NO COOKIES are being sent to the server")
    print()
    
    print("POSSIBLE CAUSES:")
    print("1. ? Domain/Path Mismatch - Cookies set for different domain/path")
    print("2. ? HTTPS/Secure Cookie Issue - Cookies require secure connection")
    print("3. ? SameSite Policy - Browser blocking cross-site cookies")
    print("4. ? Cookie Expiration - Cookies have expired")
    print("5. ?? Cookie Storage - Cookies stored in different browser profile/incognito")
    print()
    
    print("SOLUTIONS:")
    print()
    
    print("1. ? CHECK YOUR BROWSER COOKIES:")
    print("   - Open browser Developer Tools (F12)")
    print("   - Go to Application/Storage tab")
    print("   - Look for cookies under 'localhost:8100' or your domain")
    print("   - Check if 'user_id' cookie exists and its value")
    print("   - Note the Domain, Path, Secure, and SameSite settings")
    print()
    
    print("2. ? TEST WITH CURL (to bypass browser policies):")
    print("   If you have a user_id cookie value, test with:")
    print(f"   curl -H 'Cookie: user_id=YOUR_COOKIE_VALUE' {base_url}/auth/session")
    print()
    
    print("3. ? FRESH LOGIN ATTEMPT:")
    print("   Let's try a fresh Google OAuth login to create a new session...")
    
    session = requests.Session()
    
    print("\n   Step 1: Starting Google OAuth flow...")
    try:
        response = session.get(f"{base_url}/auth/google")
        if response.status_code == 302:
            google_url = response.headers.get('Location')
            print(f"   ? Redirect URL: {google_url[:100]}...")
            print("   ? MANUAL STEP: Copy this URL and complete Google login in browser")
            print("   ? Google OAuth URL:", google_url)
            print()
            print("   After completing Google login, you should be redirected back")
            print("   to your application with a session cookie set.")
        else:
            print(f"   ? Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ? Error: {e}")
    
    print("\n4. ?? ALTERNATIVE: MANUAL COOKIE CREATION")
    print("   If Google OAuth doesn't work, we can create a session manually:")
    
    # Create a test login endpoint call
    print("\n   Testing email/password login (if you have credentials)...")
    print("   You can also try creating an account first:")
    
    test_user = {
        "email": "test@example.com",
        "password": "testpassword123",
        "username": "Test User"
    }
    
    print(f"\n   Example registration:")
    print(f"   curl -X POST -H 'Content-Type: application/json' \\")
    print(f"        -d '{json.dumps(test_user)}' \\")
    print(f"        {base_url}/auth/register")
    
    login_data = {"email": test_user["email"], "password": test_user["password"]}
    print(f"\n   Example login:")
    print(f"   curl -X POST -H 'Content-Type: application/json' \\")
    print(f"        -d '{json.dumps(login_data)}' \\")
    print(f"        -c cookies.txt \\")
    print(f"        {base_url}/auth/login")
    
    print(f"\n   Then test session with saved cookies:")
    print(f"   curl -b cookies.txt {base_url}/auth/session")
    
    print("\n5. ? CONFIGURATION CHECK:")
    cookie_secret = os.environ.get('AUTH_SECRET_KEY', 'default_secret_key_change_in_production')
    print(f"   Cookie secret (first 10 chars): {cookie_secret[:10]}...")
    
    if cookie_secret == 'default_secret_key_change_in_production':
        print("   ??  WARNING: Using default cookie secret!")
        print("   Consider setting AUTH_SECRET_KEY in your .env file")
    
    print("\n6. ? CURRENT SERVER STATUS:")
    try:
        response = session.get(f"{base_url}/auth/session")
        print(f"   Session check: {response.status_code} - {response.json()}")
        
        # Check if any cookies were set
        if session.cookies:
            print("   Cookies received from server:")
            for cookie in session.cookies:
                print(f"   - {cookie.name}: {cookie.value[:20]}...")
        else:
            print("   No cookies received from server")
            
    except Exception as e:
        print(f"   ? Error checking server: {e}")
    
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("1. Check your browser cookies as described above")
    print("2. Try the manual curl commands to test")
    print("3. If needed, complete a fresh Google OAuth login")
    print("4. Report back what you find!")

if __name__ == "__main__":
    asyncio.run(fix_session_issue())
