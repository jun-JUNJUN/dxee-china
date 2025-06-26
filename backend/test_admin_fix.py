#!/usr/bin/env python3
"""
Test script to verify the admin access fix for jsakurai@acrymate.com
"""
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.service.mongodb_service import MongoDBService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_admin_fix():
    """Test the admin access fix"""
    
    # Check environment variables
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@dxee.work')
    print(f"🔍 ADMIN_EMAIL environment variable: {admin_email}")
    
    # Initialize MongoDB service
    mongodb = MongoDBService()
    
    try:
        # Look for the user with email 'jsakurai@acrymate.com'
        target_email = 'jsakurai@acrymate.com'
        print(f"\n🔍 Looking for user with email: {target_email}")
        
        # Get user by email and auth type
        user = await mongodb.get_user_by_email_and_auth_type(target_email, 'email')
        
        if not user:
            print(f"❌ User not found with email {target_email} and auth_type 'email'")
            return
        
        print(f"✅ User found:")
        print(f"   - ID: {user.get('_id')}")
        print(f"   - Email Masked: {user.get('email_masked')}")
        print(f"   - Email Hash: {user.get('email_hash')}")
        print(f"   - Auth Type: {user.get('auth_type')}")
        print(f"   - Username: {user.get('username')}")
        print(f"   - Is Verified: {user.get('is_verified')}")
        
        # Test the new admin check logic using email hashes
        print(f"\n🔍 Admin check logic (using email hashes for security):")
        
        # Generate admin email hash
        admin_email_hash = mongodb._generate_email_hash(admin_email, 'email')
        user_email_hash = user.get('email_hash')
        
        print(f"   - Admin email: {admin_email}")
        print(f"   - Admin email hash: {admin_email_hash}")
        print(f"   - User email hash: {user_email_hash}")
        print(f"   - Hashes match: {user_email_hash == admin_email_hash}")
        
        # Create a mock admin handler to test the new logic
        class MockAdminHandler:
            def __init__(self):
                self.application = type('obj', (object,), {'mongodb': mongodb})()
            
            @property
            def ADMIN_USERS(self):
                """Get admin users from environment variable (as email hashes for secure comparison)"""
                admin_email = os.environ.get('ADMIN_EMAIL', 'admin@deepschina.com')
                # Return the email hash for secure comparison with user.email_hash
                admin_email_hash = mongodb._generate_email_hash(admin_email, 'email')
                return {admin_email_hash}
            
            async def is_admin_for_user(self, user):
                """Check if given user is an admin"""
                if not user:
                    return False
                
                user_email_hash = user.get('email_hash')
                admin_email_hashes = self.ADMIN_USERS
                
                print(f"   - User email hash: {user_email_hash}")
                print(f"   - Admin email hashes: {admin_email_hashes}")
                
                is_admin = user_email_hash in admin_email_hashes
                print(f"   - Is admin: {is_admin}")
                return is_admin
        
        mock_handler = MockAdminHandler()
        is_admin = await mock_handler.is_admin_for_user(user)
        
        if is_admin:
            print(f"\n✅ SUCCESS: User {target_email} now has admin access!")
            print(f"💡 The fix works: Admin check now compares email hashes securely")
        else:
            print(f"\n❌ ISSUE: User {target_email} still does not have admin access")
            print(f"🔍 Debug info:")
            print(f"   - User email hash: {user_email_hash}")
            print(f"   - Expected admin hash: {admin_email_hash}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_admin_fix())
