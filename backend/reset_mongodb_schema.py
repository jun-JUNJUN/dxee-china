#!/usr/bin/env python3
"""
Script to reset MongoDB schema and create new privacy-focused schema
"""
import asyncio
import os
import sys
import logging
from app.service.mongodb_service import MongoDBService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_mongodb_schema():
    """
    Reset MongoDB schema completely and create new one
    """
    try:
        # Initialize MongoDB service
        mongodb = MongoDBService()
        
        print("??  Resetting MongoDB schema...")
        
        # Drop entire database to start fresh
        await mongodb.db.drop_collection("users")
        await mongodb.db.drop_collection("chats") 
        await mongodb.db.drop_collection("messages")
        
        print("? Dropped all existing collections")
        
        # Create new indexes with privacy-focused schema
        await mongodb.create_indexes()
        
        print("? Created new indexes with privacy-focused schema")
        
        # Verify the new schema
        collections = await mongodb.db.list_collection_names()
        print(f"? Available collections: {collections}")
        
        # Show indexes for users collection
        if "users" in collections:
            indexes = await mongodb.users.list_indexes().to_list(length=None)
            print("? Users collection indexes:")
            for idx in indexes:
                print(f"   - {idx['name']}: {idx.get('key', {})}")
        
        print("\n? MongoDB schema reset completed successfully!")
        print("\n? New Schema Summary:")
        print("   - user_uid: Unique user identifier (UUID)")
        print("   - email_masked: Masked email for privacy (e.g., j***doe@goo***.com)")
        print("   - email_hash: SHA256 hash of email+auth_type")
        print("   - username: Display name")
        print("   - auth_type: Authentication method (google, github, microsoft, email)")
        print("   - Provider IDs: google_id, github_id, microsoft_id")
        print("   - Timestamps: created_at, last_login, updated_at")
        
    except Exception as e:
        logger.error(f"? Error resetting MongoDB schema: {e}")
        raise
    finally:
        # Close MongoDB connection
        if mongodb._client:
            mongodb._client.close()

if __name__ == "__main__":
    try:
        asyncio.run(reset_mongodb_schema())
        print("\n? You can now restart your server to use the new schema!")
    except Exception as e:
        print(f"? Schema reset failed: {e}")
        sys.exit(1)
