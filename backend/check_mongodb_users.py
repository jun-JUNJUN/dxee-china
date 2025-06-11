#!/usr/bin/env python3
"""
Script to check MongoDB users
"""
import asyncio
import os
from dotenv import load_dotenv
from app.service.mongodb_service import MongoDBService

async def check_users():
    """Check users in MongoDB"""
    print("🔍 Checking MongoDB users...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize MongoDB service
    mongodb = MongoDBService()
    
    try:
        # Get all users
        users_collection = mongodb.db.users
        users = await users_collection.find({}).to_list(length=None)
        
        print(f"📊 Total users found: {len(users)}")
        print()
        
        if users:
            for i, user in enumerate(users, 1):
                print(f"👤 User {i}:")
                print(f"   ID: {user.get('_id')}")
                print(f"   Email: {user.get('email')}")
                print(f"   Username: {user.get('username')}")
                print(f"   Auth Type: {user.get('auth_type')}")
                print(f"   Google ID: {user.get('google_id')}")
                print(f"   Created: {user.get('created_at')}")
                print(f"   Last Login: {user.get('last_login')}")
                print("-" * 30)
        else:
            print("❌ No users found in the database")
            
    except Exception as e:
        print(f"❌ Error checking users: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the MongoDB connection
        if hasattr(mongodb, 'client'):
            mongodb.client.close()

if __name__ == "__main__":
    asyncio.run(check_users())
