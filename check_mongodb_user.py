#!/usr/bin/env python3
"""
Script to check if administrator account exists in MongoDB
Usage: ./activate_backend.sh && uv run python check_mongodb_user.py
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def check_user_in_mongodb():
    """Check if the administrator account exists in MongoDB"""
    
    # Get MongoDB URI from environment
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(mongodb_uri)
    
    try:
        # Get database (assuming it's named after the project or default)
        # Check what databases exist first
        db_names = await client.list_database_names()
        print(f"Available databases: {db_names}")
        
        # Try common database names (including ones found: dxeechina, superdx)
        possible_db_names = ['dxeechina', 'superdx', 'deepschina', 'dxee_china', 'dxee-china', 'test', 'admin']
        
        target_email = "jsakurai@acrymate.com"
        
        for db_name in possible_db_names:
            if db_name in db_names:
                db = client[db_name]
                collections = await db.list_collection_names()
                print(f"\nDatabase '{db_name}' collections: {collections}")
                
                if 'users' in collections:
                    users_collection = db.users
                    
                    # Search for the specific email
                    user = await users_collection.find_one({"email": target_email})
                    if user:
                        print(f"\n✓ Found user in database '{db_name}':")
                        print(f"  Email: {user.get('email')}")
                        print(f"  Verified: {user.get('verified', 'Not set')}")
                        print(f"  Created: {user.get('created_at', 'Not set')}")
                        print(f"  ID: {user.get('_id')}")
                        
                        # Check all fields
                        print(f"  All fields: {list(user.keys())}")
                        return user
                    
                    # Also search for similar emails (case insensitive)
                    similar_users = []
                    async for user in users_collection.find({"email": {"$regex": "jsakurai", "$options": "i"}}):
                        similar_users.append(user)
                    
                    if similar_users:
                        print(f"\n~ Found similar users in database '{db_name}':")
                        for user in similar_users:
                            print(f"  Email: {user.get('email')}")
                            print(f"  Verified: {user.get('verified', 'Not set')}")
        
        print(f"\n✗ User '{target_email}' not found in any database")
        
        # Show all users for debugging
        for db_name in possible_db_names:
            if db_name in db_names:
                db = client[db_name]
                if 'users' in await db.list_collection_names():
                    users_collection = db.users
                    user_count = await users_collection.count_documents({})
                    print(f"\nDatabase '{db_name}' has {user_count} total users")
                    
                    if user_count > 0 and user_count <= 20:  # Only show if manageable number
                        print("All users:")
                        async for user in users_collection.find({}):
                            print(f"  - Email: {user.get('email', 'Not set')}")
                            print(f"    Verified: {user.get('verified', user.get('is_verified', 'Not set'))}")
                            print(f"    Created: {user.get('created_at', 'Not set')}")
                            print(f"    User ID: {user.get('_id')}")
                            print(f"    All fields: {list(user.keys())}")
                            print("    ---")
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(check_user_in_mongodb())