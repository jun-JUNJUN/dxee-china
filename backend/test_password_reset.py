#!/usr/bin/env python3
"""
Test script for password reset functionality
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.service.mongodb_service import MongoDBService
from app.handler.auth_handler import send_password_reset_email
import bcrypt
from datetime import datetime, timedelta

async def test_password_reset():
    """Test the password reset functionality"""
    print("? Testing Password Reset Functionality")
    print("=" * 50)
    
    # Initialize MongoDB service
    mongodb = MongoDBService()
    
    try:
        # Create indexes
        await mongodb.create_indexes()
        print("? MongoDB indexes created")
        
        # Test user data
        test_email = "test@example.com"
        test_password = "testpassword123"
        
        # Create a test user
        hashed_password = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt())
        user_doc = {
            'email': test_email,
            'username': 'testuser',
            'password': hashed_password.decode('utf-8'),
            'auth_type': 'email',
            'is_verified': True,
            'email_verified_at': datetime.utcnow()
        }
        
        user_id = await mongodb.create_user(user_doc)
        print(f"? Test user created with ID: {user_id}")
        
        # Test rate limiting check
        rate_limit_ok = await mongodb.check_reset_rate_limit(test_email, 'email')
        print(f"? Rate limit check: {'PASSED' if rate_limit_ok else 'FAILED'}")
        
        # Test reset token generation and storage
        import secrets
        reset_token = secrets.token_urlsafe(24)[:32]
        reset_token_expires = datetime.utcnow() + timedelta(minutes=30)
        
        token_stored = await mongodb.update_user_reset_token(
            test_email, 'email', reset_token, reset_token_expires
        )
        print(f"? Reset token storage: {'SUCCESS' if token_stored else 'FAILED'}")
        
        # Test token retrieval
        user_by_token = await mongodb.get_user_by_reset_token(reset_token)
        print(f"? Token retrieval: {'SUCCESS' if user_by_token else 'FAILED'}")
        
        # Test email sending (will log the reset URL since SMTP might not be configured)
        email_sent = await send_password_reset_email(test_email, reset_token)
        print(f"? Email sending: {'SUCCESS' if email_sent else 'FAILED'}")
        
        # Test password reset
        new_password = "newpassword456"
        new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        
        update_result = await mongodb.update_user(str(user_id), {
            'password': new_hashed_password.decode('utf-8')
        })
        print(f"? Password update: {'SUCCESS' if update_result > 0 else 'FAILED'}")
        
        # Test token cleanup
        token_cleared = await mongodb.clear_reset_token(str(user_id))
        print(f"? Token cleanup: {'SUCCESS' if token_cleared else 'FAILED'}")
        
        # Verify token is no longer valid
        user_by_token_after = await mongodb.get_user_by_reset_token(reset_token)
        print(f"? Token invalidation: {'SUCCESS' if not user_by_token_after else 'FAILED'}")
        
        print("\n? All password reset tests completed!")
        
    except Exception as e:
        print(f"? Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup: remove test user
        try:
            if 'user_id' in locals():
                await mongodb.users.delete_one({"_id": user_id})
                print("? Test user cleaned up")
        except Exception as e:
            print(f"?? Cleanup warning: {e}")

if __name__ == "__main__":
    asyncio.run(test_password_reset())
