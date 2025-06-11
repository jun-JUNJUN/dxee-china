#!/usr/bin/env python3
"""
Test script for Google OAuth configuration
"""
import os
import sys
from dotenv import load_dotenv

def test_google_auth_config():
    """Test Google OAuth configuration"""
    print("? Testing Google OAuth Configuration...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    google_client_id = os.environ.get('GOOGLE_CLIENT_ID')
    google_client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
    google_redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI')
    port = os.environ.get('PORT', '8100')
    
    print(f"? Configuration Check:")
    print(f"   PORT: {port}")
    print(f"   GOOGLE_CLIENT_ID: {'? Set' if google_client_id else '? Missing'}")
    print(f"   GOOGLE_CLIENT_SECRET: {'? Set' if google_client_secret else '? Missing'}")
    print(f"   GOOGLE_REDIRECT_URI: {google_redirect_uri or '? Missing'}")
    
    print(f"\n? Expected Configuration:")
    print(f"   Server should run on: http://localhost:{port}")
    print(f"   Google Auth Platform should have:")
    print(f"   - Authorized JavaScript origins: http://localhost:{port}")
    print(f"   - Authorized redirect URIs: http://localhost:{port}/auth/google/callback")
    
    # Check if all required variables are set
    missing_vars = []
    if not google_client_id:
        missing_vars.append('GOOGLE_CLIENT_ID')
    if not google_client_secret:
        missing_vars.append('GOOGLE_CLIENT_SECRET')
    if not google_redirect_uri:
        missing_vars.append('GOOGLE_REDIRECT_URI')
    
    if missing_vars:
        print(f"\n? Missing environment variables: {', '.join(missing_vars)}")
        print(f"\n? To fix this, update your .env file with:")
        for var in missing_vars:
            if var == 'GOOGLE_CLIENT_ID':
                print(f"   {var}=your_google_client_id_here")
            elif var == 'GOOGLE_CLIENT_SECRET':
                print(f"   {var}=your_google_client_secret_here")
            elif var == 'GOOGLE_REDIRECT_URI':
                print(f"   {var}=http://localhost:{port}/auth/google/callback")
        return False
    else:
        print(f"\n? All environment variables are set!")
        print(f"\n? Ready to test Google OAuth!")
        print(f"\n? Next steps:")
        print(f"   1. Make sure your Google Auth Platform is configured correctly")
        print(f"   2. Start the server: python backend/app/tornado_main.py")
        print(f"   3. Open browser: http://localhost:{port}")
        print(f"   4. Click on the Google Sign-In button")
        return True

if __name__ == "__main__":
    success = test_google_auth_config()
    sys.exit(0 if success else 1)
