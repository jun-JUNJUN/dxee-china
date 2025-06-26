#!/usr/bin/env python3
"""
Test script to check admin access for jsakurai@acrymate.com
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
from app.handler.admin_handler import AdminHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_admin_access():
    """Test admin access for the user"""
    
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
        print(f"   - Auth Type: {user.get('auth_type')}")
        print(f"   - Username: {user.get('username')}")
        print(f"   - Is Verified: {user.get('is_verified')}")
        
        # Check what the masked email looks like
        masked_email = mongodb._mask_email(target_email)
        admin_masked = mongodb._mask_email(admin_email)
        
        print(f"\n🎭 Email masking comparison:")
        print(f"   - Target email: {target_email}")
        print(f"   - Target masked: {masked_email}")
        print(f"   - Admin email: {admin_email}")
        print(f"   - Admin masked: {admin_masked}")
        print(f"   - Match: {masked_email == admin_masked}")
        
        # Test admin check logic
        print(f"\n🔍 Admin check logic:")
        
        # Create a mock admin handler to test the logic
        class MockAdminHandler:
            def __init__(self):
                self.application = type('obj', (object,), {'mongodb': mongodb})()
            
            @property
            def ADMIN_USERS(self):
                """Get admin users from environment variable"""
                admin_email = os.environ.get('ADMIN_EMAIL', 'admin@dxee.work')
                return {admin_email}
            
            async def is_admin_for_user(self, user):
                """Check if given user is an admin"""
                if not user:
                    return False
                
                user_email = user.get('email_masked')
                admin_emails = self.ADMIN_USERS
                
                print(f"   - User email masked: {user_email}")
                print(f"   - Admin emails: {admin_emails}")
                
                is_admin = user_email in admin_emails
                print(f"   - Is admin: {is_admin}")
                return is_admin
        
        mock_handler = MockAdminHandler()
        is_admin = await mock_handler.is_admin_for_user(user)
        
        if is_admin:
            print(f"\n✅ SUCCESS: User {target_email} should have admin access!")
        else:
            print(f"\n❌ ISSUE: User {target_email} does not have admin access")
            print(f"💡 SOLUTION: The issue is that the admin check compares:")
            print(f"   - user.email_masked ({user.get('email_masked')})")
            print(f"   - with ADMIN_EMAIL ({admin_email})")
            print(f"   - But it should compare with the masked version of ADMIN_EMAIL ({admin_masked})")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_admin_access())
