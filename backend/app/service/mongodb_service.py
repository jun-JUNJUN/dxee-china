import os
import logging
import motor.motor_tornado
import hashlib
import uuid
from datetime import datetime, timedelta
from bson import ObjectId

# Get logger
logger = logging.getLogger(__name__)

class MongoDBService:
    """
    Service for interacting with MongoDB
    """
    def __init__(self, io_loop=None):
        self.mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
        self.db_name = os.environ.get('MONGODB_DB_NAME', 'dxeechina')
        self.io_loop = io_loop
        
        # Defer client initialization until first use
        self._client = None
        self._db = None
        self._users = None
        self._chats = None
        self._messages = None
        
        logger.info(f"MongoDB service configured with URI: {self.mongodb_uri}")
    
    def _mask_email(self, email):
        """
        Mask email address for privacy
        Example: john.doe@example.com -> j***doe@exa***.com
        Pattern: first char + *** + last 3 chars of local @ first 3 chars + *** + domain extension
        """
        if not email or '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        
        # Handle domain parts (separate main domain from extension)
        domain_parts = domain.split('.')
        if len(domain_parts) < 2:
            return email  # Invalid email format
        
        main_domain = domain_parts[0]
        domain_extension = '.'.join(domain_parts[1:])  # .com, .co.uk, etc.
        
        # Mask local part: first char + *** + last 3 chars
        if len(local) <= 4:
            # For short local parts, just show first char + ***
            masked_local = local[0] + '***'
        else:
            # Show first char + *** + last 3 chars
            masked_local = local[0] + '***' + local[-3:]
        
        # Mask domain: first 3 chars + *** + extension
        if len(main_domain) <= 3:
            # For short domains, just show first char + ***
            masked_domain = main_domain[0] + '***'
        else:
            # Show first 3 chars + ***
            masked_domain = main_domain[:3] + '***'
        
        return f"{masked_local}@{masked_domain}.{domain_extension}"
    
    def _generate_email_hash(self, email, auth_type):
        """
        Generate a unique hash for email + auth_type combination
        """
        combined = f"{email.lower()}:{auth_type}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _generate_user_uid(self):
        """
        Generate a unique user identifier
        """
        return str(uuid.uuid4())
    
    def _ensure_client(self):
        """
        Ensure the MongoDB client is initialized with the current IOLoop
        """
        if self._client is None:
            import tornado.ioloop
            
            # Use the provided io_loop or get the current one
            current_loop = self.io_loop or tornado.ioloop.IOLoop.current()
            
            logger.info(f"Initializing MongoDB client with URI: {self.mongodb_uri}")
            
            # Initialize MongoDB client
            self._client = motor.motor_tornado.MotorClient(
                self.mongodb_uri,
                io_loop=current_loop
            )
            self._db = self._client[self.db_name]
            
            # Define collections
            self._users = self._db.users
            self._chats = self._db.chats
            self._messages = self._db.messages
            
            logger.info("MongoDB client initialized")
    
    @property
    def client(self):
        self._ensure_client()
        return self._client
    
    @property
    def db(self):
        self._ensure_client()
        return self._db
    
    @property
    def users(self):
        self._ensure_client()
        return self._users
    
    @property
    def chats(self):
        self._ensure_client()
        return self._chats
    
    @property
    def messages(self):
        self._ensure_client()
        return self._messages
    
    async def create_indexes(self):
        """
        Create necessary indexes for the collections
        """
        try:
            # Drop existing indexes and clear users collection
            try:
                await self.users.drop_index("email_1")
                logger.info("Dropped existing unique email index")
            except Exception:
                pass  # Index might not exist
            
            # Clear existing users to avoid conflicts with new schema
            result = await self.users.delete_many({})
            logger.info(f"Cleared {result.deleted_count} existing users for schema update")
            
            # Create indexes for users collection with new schema
            await self.users.create_index("user_uid", unique=True)  # Unique user identifier
            await self.users.create_index("email_hash", unique=True)  # Unique hash for email+auth_type
            await self.users.create_index("email_masked")  # Index for masked email lookups
            await self.users.create_index("username")
            await self.users.create_index("auth_type")
            await self.users.create_index([("email_masked", 1), ("auth_type", 1)])  # Compound index
            
            # Password reset indexes
            await self.users.create_index("reset_token", sparse=True)  # Index for reset token lookups
            await self.users.create_index("reset_token_expires", sparse=True)  # Index for expiration cleanup
            
            # Create indexes for chats collection
            await self.chats.create_index("user_id")
            await self.chats.create_index("updated_at")
            
            # Create indexes for messages collection
            await self.messages.create_index([("chat_id", 1), ("timestamp", 1)])
            await self.messages.create_index([("user_id", 1), ("chat_id", 1)])
            await self.messages.create_index("shared", sparse=True)
            
            logger.info("MongoDB indexes created successfully with new privacy-focused schema")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
            raise
    
    # User methods
    
    async def create_user(self, user_data):
        """
        Create a new user with privacy-focused schema
        """
        try:
            # Generate unique identifiers
            user_data["user_uid"] = self._generate_user_uid()
            
            # Handle email masking and hashing
            if "email" in user_data:
                email = user_data["email"]
                auth_type = user_data.get("auth_type", "email")
                
                # Store masked email instead of real email
                user_data["email_masked"] = self._mask_email(email)
                user_data["email_hash"] = self._generate_email_hash(email, auth_type)
                
                # Remove the original email for privacy
                del user_data["email"]
            
            user_data["created_at"] = datetime.utcnow()
            result = await self.users.insert_one(user_data)
            logger.info(f"User created with ID: {result.inserted_id}, UID: {user_data['user_uid']}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    async def get_user_by_id(self, user_id):
        """
        Get a user by ID
        """
        try:
            user = await self.users.find_one({"_id": ObjectId(user_id)})
            return user
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            raise
    
    async def get_user_by_email(self, email, auth_type=None):
        """
        Get a user by email (using email hash for privacy)
        """
        try:
            if auth_type:
                # Find by specific auth_type
                email_hash = self._generate_email_hash(email, auth_type)
                user = await self.users.find_one({"email_hash": email_hash})
            else:
                # Find any user with this email (first match)
                email_masked = self._mask_email(email)
                user = await self.users.find_one({"email_masked": email_masked})
            return user
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            raise
    
    async def get_user_by_email_and_auth_type(self, email, auth_type):
        """
        Get a user by email and auth_type using email hash
        """
        try:
            email_hash = self._generate_email_hash(email, auth_type)
            user = await self.users.find_one({"email_hash": email_hash})
            return user
        except Exception as e:
            logger.error(f"Error getting user by email and auth_type: {e}")
            raise
    
    async def get_users_by_email_masked(self, email):
        """
        Get all users with the same masked email (different auth_types)
        """
        try:
            email_masked = self._mask_email(email)
            cursor = self.users.find({"email_masked": email_masked})
            users = []
            async for user in cursor:
                users.append(user)
            return users
        except Exception as e:
            logger.error(f"Error getting users by masked email: {e}")
            raise
    
    async def get_user_by_uid(self, user_uid):
        """
        Get a user by unique user identifier
        """
        try:
            user = await self.users.find_one({"user_uid": user_uid})
            return user
        except Exception as e:
            logger.error(f"Error getting user by UID: {e}")
            raise
    
    async def update_user(self, user_id, update_data):
        """
        Update a user
        """
        try:
            update_data["updated_at"] = datetime.utcnow()
            result = await self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            logger.info(f"User updated: {result.modified_count} document(s) modified")
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            raise
    
    # Chat methods
    
    async def create_chat(self, chat_data):
        """
        Create a new chat
        """
        try:
            chat_data["created_at"] = datetime.utcnow()
            chat_data["updated_at"] = datetime.utcnow()
            result = await self.chats.insert_one(chat_data)
            logger.info(f"Chat created with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error creating chat: {e}")
            raise
    
    async def get_chat_by_id(self, chat_id):
        """
        Get a chat by ID
        """
        try:
            if isinstance(chat_id, str) and len(chat_id) == 24:
                # If it's a valid ObjectId string
                chat = await self.chats.find_one({"_id": ObjectId(chat_id)})
            else:
                # If it's a UUID string from the old system
                chat = await self.chats.find_one({"chat_id": chat_id})
            
            return chat
        except Exception as e:
            logger.error(f"Error getting chat by ID: {e}")
            raise
    
    async def get_user_chats(self, user_id, limit=20, skip=0):
        """
        Get chats for a user
        """
        try:
            cursor = self.chats.find({"user_id": user_id}) \
                .sort("updated_at", -1) \
                .skip(skip) \
                .limit(limit)
            
            chats = []
            async for chat in cursor:
                chats.append(chat)
            
            return chats
        except Exception as e:
            logger.error(f"Error getting user chats: {e}")
            raise
    
    async def update_chat(self, chat_id, update_data):
        """
        Update a chat
        """
        try:
            update_data["updated_at"] = datetime.utcnow()
            
            if isinstance(chat_id, str) and len(chat_id) == 24:
                # If it's a valid ObjectId string
                result = await self.chats.update_one(
                    {"_id": ObjectId(chat_id)},
                    {"$set": update_data}
                )
            else:
                # If it's a UUID string from the old system
                result = await self.chats.update_one(
                    {"chat_id": chat_id},
                    {"$set": update_data}
                )
            
            logger.info(f"Chat updated: {result.modified_count} document(s) modified")
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating chat: {e}")
            raise
    
    # Message methods
    
    async def create_message(self, message_data):
        """
        Create a new message
        """
        try:
            message_data["created_at"] = datetime.utcnow()
            result = await self.messages.insert_one(message_data)
            
            # Update the chat's updated_at timestamp
            await self.update_chat(message_data["chat_id"], {"updated_at": datetime.utcnow()})
            
            logger.info(f"Message created with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise
    
    async def get_chat_messages(self, chat_id, limit=100, skip=0):
        """
        Get messages for a chat
        """
        try:
            cursor = self.messages.find({"chat_id": chat_id}) \
                .sort("timestamp", 1) \
                .skip(skip) \
                .limit(limit)
            
            messages = []
            async for message in cursor:
                messages.append(message)
            
            return messages
        except Exception as e:
            logger.error(f"Error getting chat messages: {e}")
            raise
    
    async def share_message(self, message_id, share=True):
        """
        Mark a message as shared or unshared
        """
        try:
            result = await self.messages.update_one(
                {"_id": ObjectId(message_id)},
                {"$set": {"shared": share, "shared_at": datetime.utcnow() if share else None}}
            )
            
            logger.info(f"Message sharing updated: {result.modified_count} document(s) modified")
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating message sharing: {e}")
            raise
    
    async def get_shared_messages(self, limit=100, skip=0):
        """
        Get all shared messages
        """
        try:
            cursor = self.messages.find({"shared": True}) \
                .sort("shared_at", -1) \
                .skip(skip) \
                .limit(limit)
            
            messages = []
            async for message in cursor:
                messages.append(message)
            
            return messages
        except Exception as e:
            logger.error(f"Error getting shared messages: {e}")
            raise
    
    # Password Reset methods
    
    async def update_user_reset_token(self, email, auth_type, reset_token, reset_token_expires):
        """
        Store password reset token for a user
        """
        try:
            import hashlib
            
            # Hash the token for secure storage
            hashed_token = hashlib.sha256(reset_token.encode()).hexdigest()
            
            # Find user by email and auth_type
            email_hash = self._generate_email_hash(email, auth_type)
            
            # Update user with reset token info
            result = await self.users.update_one(
                {"email_hash": email_hash},
                {
                    "$set": {
                        "reset_token": hashed_token,
                        "reset_token_expires": reset_token_expires,
                        "reset_token_created": datetime.utcnow()
                    },
                    "$inc": {"reset_attempts": 1}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Reset token updated for user with email hash: {email_hash[:8]}...")
                return True
            else:
                logger.warning(f"No user found to update reset token for email hash: {email_hash[:8]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error updating user reset token: {e}")
            raise
    
    async def get_user_by_reset_token(self, reset_token):
        """
        Get user by reset token (validates token and expiration)
        """
        try:
            import hashlib
            
            # Hash the provided token to match stored hash
            hashed_token = hashlib.sha256(reset_token.encode()).hexdigest()
            
            # Find user with matching token that hasn't expired
            user = await self.users.find_one({
                "reset_token": hashed_token,
                "reset_token_expires": {"$gt": datetime.utcnow()}
            })
            
            if user:
                logger.info(f"Valid reset token found for user: {user.get('user_uid', 'unknown')}")
            else:
                logger.warning("Invalid or expired reset token provided")
            
            return user
            
        except Exception as e:
            logger.error(f"Error getting user by reset token: {e}")
            raise
    
    async def clear_reset_token(self, user_id):
        """
        Clear password reset token after successful reset
        """
        try:
            result = await self.users.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$unset": {
                        "reset_token": "",
                        "reset_token_expires": "",
                        "reset_token_created": ""
                    },
                    "$set": {
                        "password_reset_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Reset token cleared for user: {user_id}")
                return True
            else:
                logger.warning(f"No user found to clear reset token: {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing reset token: {e}")
            raise
    
    async def check_reset_rate_limit(self, email, auth_type, max_attempts=3, time_window_hours=1):
        """
        Check if user has exceeded reset request rate limit
        """
        try:
            email_hash = self._generate_email_hash(email, auth_type)
            
            # Find user and check reset attempts
            user = await self.users.find_one({"email_hash": email_hash})
            
            if not user:
                return True  # Allow if user doesn't exist (don't reveal existence)
            
            reset_attempts = user.get('reset_attempts', 0)
            last_reset_request = user.get('reset_token_created')
            
            # If no previous attempts, allow
            if reset_attempts == 0 or not last_reset_request:
                return True
            
            # Check if time window has passed
            time_diff = datetime.utcnow() - last_reset_request
            if time_diff.total_seconds() > (time_window_hours * 3600):
                # Reset the counter if time window has passed
                await self.users.update_one(
                    {"email_hash": email_hash},
                    {"$set": {"reset_attempts": 0}}
                )
                return True
            
            # Check if under rate limit
            if reset_attempts < max_attempts:
                return True
            
            logger.warning(f"Rate limit exceeded for email hash: {email_hash[:8]}...")
            return False
            
        except Exception as e:
            logger.error(f"Error checking reset rate limit: {e}")
            raise
    
    # Enhanced Cache Service Methods for DeepSeek Research
    
    @property
    def web_content_cache(self):
        """Get web content cache collection"""
        self._ensure_client()
        return self._db.web_content_cache
    
    @property
    def research_sessions(self):
        """Get research sessions collection"""
        self._ensure_client()
        return self._db.research_sessions
    
    @property
    def api_usage_logs(self):
        """Get API usage logs collection"""
        self._ensure_client()
        return self._db.api_usage_logs
    
    @property
    def deepthink_cache(self):
        """Get deep-think content cache collection"""
        self._ensure_client()
        return self._db.deepthink_cache
    
    @property
    def deepthink_results(self):
        """Get deep-think results collection"""
        self._ensure_client()
        return self._db.deepthink_results
    
    async def create_research_indexes(self, cache_expiry_days=30):
        """
        Create indexes for research-related collections
        
        Args:
            cache_expiry_days: TTL for cache entries in days
        """
        try:
            # Web content cache indexes
            await self.web_content_cache.create_index("url", unique=True)
            await self.web_content_cache.create_index("keywords")
            await self.web_content_cache.create_index(
                "created_at", 
                expireAfterSeconds=cache_expiry_days * 24 * 3600  # TTL in seconds
            )
            await self.web_content_cache.create_index("access_count")
            
            # Research sessions indexes
            await self.research_sessions.create_index("session_id", unique=True)
            await self.research_sessions.create_index("chat_id")
            await self.research_sessions.create_index("created_at")
            await self.research_sessions.create_index("user_id")
            
            # API usage logs indexes
            await self.api_usage_logs.create_index("timestamp")
            await self.api_usage_logs.create_index("api_type")
            await self.api_usage_logs.create_index("chat_id")
            await self.api_usage_logs.create_index("success")
            
            logger.info(f"Research indexes created successfully with {cache_expiry_days} day TTL")
            
        except Exception as e:
            logger.error(f"Error creating research indexes: {e}")
            raise
    
    async def create_deepthink_indexes(self, cache_expiry_days=30):
        """
        Create indexes for deep-think specific collections
        
        Args:
            cache_expiry_days: TTL for cache entries in days
        """
        try:
            # Deep-think cache indexes
            await self._create_index_safe(self.deepthink_cache, "content_hash", unique=True)
            await self._create_index_safe(self.deepthink_cache, "url")
            await self._create_index_safe(self.deepthink_cache, "request_id")
            # Text index for content search - create with safe fallback
            try:
                await self.deepthink_cache.create_index([("content", "text"), ("query_text", "text")])
            except Exception as e:
                logger.debug(f"Text index creation failed (may already exist): {e}")
            await self._create_index_safe(self.deepthink_cache, "hit_count")
            await self._create_index_safe(self.deepthink_cache, "relevance_score")
            
            # TTL index with unique name for cache
            await self._create_ttl_index_safe(
                self.deepthink_cache, 
                "created_at", 
                "deepthink_cache_ttl",
                cache_expiry_days * 24 * 3600
            )
            
            # Deep-think results indexes  
            await self._create_index_safe(self.deepthink_results, "request_id", unique=True)
            await self._create_index_safe(self.deepthink_results, "chat_id")
            await self._create_index_safe(self.deepthink_results, "user_id")
            await self._create_index_safe(self.deepthink_results, "created_at")
            await self._create_index_safe(self.deepthink_results, "confidence")
            await self._create_index_safe(self.deepthink_results, "execution_time")
            
            # TTL index with unique name for results
            await self._create_ttl_index_safe(
                self.deepthink_results,
                "created_at", 
                "deepthink_results_ttl",
                cache_expiry_days * 24 * 3600
            )
            
            logger.info(f"Deep-think indexes created successfully with {cache_expiry_days} day TTL")
            
        except Exception as e:
            logger.error(f"Error creating deep-think indexes: {e}")
            raise
    
    async def _create_index_safe(self, collection, keys, **kwargs):
        """
        Create index safely, ignoring duplicate key errors
        """
        try:
            await collection.create_index(keys, **kwargs)
        except Exception as e:
            if "already exists" in str(e) or "duplicate key" in str(e).lower():
                logger.debug(f"Index already exists (ignoring): {keys}")
            else:
                raise
    
    async def _create_ttl_index_safe(self, collection, field, index_name, expire_after_seconds):
        """
        Create TTL index safely, handling naming conflicts
        """
        try:
            await collection.create_index(
                field,
                name=index_name,
                expireAfterSeconds=expire_after_seconds
            )
        except Exception as e:
            if "already exists" in str(e) or "IndexOptionsConflict" in str(e):
                logger.debug(f"TTL index conflict for {index_name}, attempting to recreate...")
                try:
                    # Drop the conflicting index and recreate
                    await collection.drop_index(index_name)
                    await collection.create_index(
                        field,
                        name=index_name,
                        expireAfterSeconds=expire_after_seconds
                    )
                    logger.info(f"TTL index recreated: {index_name}")
                except Exception as recreate_error:
                    logger.warning(f"Could not recreate TTL index {index_name}: {recreate_error}")
            else:
                raise
    
    # Web Content Cache Methods
    
    async def get_cached_content(self, url: str):
        """
        Get cached content for URL
        
        Args:
            url: The URL to lookup
            
        Returns:
            Cached content data if found and fresh, None otherwise
        """
        try:
            cached = await self.web_content_cache.find_one({'url': url})
            if cached:
                # Increment access count
                await self.web_content_cache.update_one(
                    {'url': url},
                    {'$inc': {'access_count': 1, 'last_accessed': datetime.utcnow()}}
                )
                return cached.get('content')
            return None
        except Exception as e:
            logger.error(f"Error getting cached content for {url}: {e}")
            return None
    
    async def cache_content(self, url: str, content: dict, keywords: list):
        """
        Cache content with keywords and metadata
        
        Args:
            url: The URL being cached
            content: Content data to cache
            keywords: Keywords associated with the content
        """
        try:
            cache_data = {
                'url': url,
                'content': content,
                'keywords': keywords,
                'created_at': datetime.utcnow(),
                'last_accessed': datetime.utcnow(),
                'access_count': 1
            }
            
            await self.web_content_cache.update_one(
                {'url': url},
                {'$set': cache_data},
                upsert=True
            )
            
            logger.info(f"Content cached successfully for URL: {url}")
            
        except Exception as e:
            logger.error(f"Error caching content for {url}: {e}")
    
    async def search_cached_content(self, keywords: list, limit: int = 50):
        """
        Search cached content by keywords
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching cached content
        """
        try:
            # Create search query using $in operator for keywords
            query = {'keywords': {'$in': keywords}}
            
            cursor = self.web_content_cache.find(query).sort('access_count', -1).limit(limit)
            results = []
            async for doc in cursor:
                results.append(doc)
            
            logger.info(f"Found {len(results)} cached content items for keywords: {keywords}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching cached content: {e}")
            return []
    
    async def get_cache_stats(self):
        """
        Get cache statistics and health metrics
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            total_entries = await self.web_content_cache.count_documents({})
            
            # Count successful vs failed entries
            successful_entries = await self.web_content_cache.count_documents({'content.success': True})
            failed_entries = total_entries - successful_entries
            
            # Get most accessed URLs
            most_accessed = []
            cursor = self.web_content_cache.find({}, {'url': 1, 'access_count': 1}).sort('access_count', -1).limit(5)
            async for doc in cursor:
                most_accessed.append({
                    'url': doc['url'],
                    'access_count': doc.get('access_count', 0)
                })
            
            # Calculate average access count
            pipeline = [
                {'$group': {'_id': None, 'avg_access': {'$avg': '$access_count'}}}
            ]
            avg_result = []
            async for doc in self.web_content_cache.aggregate(pipeline):
                avg_result.append(doc)
            
            avg_access = avg_result[0]['avg_access'] if avg_result else 0
            
            return {
                'total_entries': total_entries,
                'successful_entries': successful_entries,
                'failed_entries': failed_entries,
                'success_rate': (successful_entries / total_entries * 100) if total_entries > 0 else 0,
                'average_access_count': round(avg_access, 2),
                'most_accessed': most_accessed
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'total_entries': 0,
                'successful_entries': 0,
                'failed_entries': 0,
                'success_rate': 0,
                'average_access_count': 0,
                'most_accessed': []
            }
    
    # Research Session Methods
    
    async def create_research_session(self, session_data):
        """
        Create a new research session
        
        Args:
            session_data: Session data including session_id, chat_id, user_id, question, etc.
            
        Returns:
            Inserted session ID
        """
        try:
            session_data['created_at'] = datetime.utcnow()
            session_data['updated_at'] = datetime.utcnow()
            session_data['status'] = 'started'
            
            result = await self.research_sessions.insert_one(session_data)
            logger.info(f"Research session created: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            logger.error(f"Error creating research session: {e}")
            raise
    
    async def update_research_session(self, session_id: str, update_data: dict):
        """
        Update research session with progress or results
        
        Args:
            session_id: Unique session identifier
            update_data: Data to update
            
        Returns:
            Number of documents modified
        """
        try:
            update_data['updated_at'] = datetime.utcnow()
            
            result = await self.research_sessions.update_one(
                {'session_id': session_id},
                {'$set': update_data}
            )
            
            logger.info(f"Research session updated: {session_id}")
            return result.modified_count
            
        except Exception as e:
            logger.error(f"Error updating research session {session_id}: {e}")
            raise
    
    async def get_research_session(self, session_id: str):
        """
        Get research session by ID
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session document or None
        """
        try:
            session = await self.research_sessions.find_one({'session_id': session_id})
            return session
        except Exception as e:
            logger.error(f"Error getting research session {session_id}: {e}")
            return None
    
    async def get_user_research_sessions(self, user_id: str, limit: int = 20):
        """
        Get research sessions for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of research sessions
        """
        try:
            cursor = self.research_sessions.find({'user_id': user_id}) \
                .sort('created_at', -1) \
                .limit(limit)
            
            sessions = []
            async for session in cursor:
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting user research sessions: {e}")
            return []
    
    # API Usage Logging Methods
    
    async def log_api_usage(self, log_data):
        """
        Log API usage for monitoring and analytics
        
        Args:
            log_data: Usage data including api_type, endpoint, response_time, success, etc.
        """
        try:
            log_data['timestamp'] = datetime.utcnow()
            await self.api_usage_logs.insert_one(log_data)
        except Exception as e:
            logger.error(f"Error logging API usage: {e}")
    
    async def get_api_usage_stats(self, hours_back: int = 24):
        """
        Get API usage statistics for the specified time period
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Dictionary with usage statistics
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Total API calls
            total_calls = await self.api_usage_logs.count_documents({
                'timestamp': {'$gte': cutoff_time}
            })
            
            # Success rate
            successful_calls = await self.api_usage_logs.count_documents({
                'timestamp': {'$gte': cutoff_time},
                'success': True
            })
            
            # API breakdown
            pipeline = [
                {'$match': {'timestamp': {'$gte': cutoff_time}}},
                {'$group': {
                    '_id': '$api_type',
                    'count': {'$sum': 1},
                    'avg_response_time': {'$avg': '$response_time'},
                    'success_rate': {
                        '$avg': {'$cond': [{'$eq': ['$success', True]}, 1, 0]}
                    }
                }},
                {'$sort': {'count': -1}}
            ]
            
            api_breakdown = []
            async for doc in self.api_usage_logs.aggregate(pipeline):
                api_breakdown.append({
                    'api_type': doc['_id'],
                    'call_count': doc['count'],
                    'avg_response_time': round(doc['avg_response_time'], 2),
                    'success_rate': round(doc['success_rate'] * 100, 1)
                })
            
            return {
                'time_period_hours': hours_back,
                'total_api_calls': total_calls,
                'successful_calls': successful_calls,
                'success_rate': round((successful_calls / total_calls * 100), 1) if total_calls > 0 else 0,
                'api_breakdown': api_breakdown
            }
            
        except Exception as e:
            logger.error(f"Error getting API usage stats: {e}")
            return {
                'time_period_hours': hours_back,
                'total_api_calls': 0,
                'successful_calls': 0,
                'success_rate': 0,
                'api_breakdown': []
            }
    
    # Deep-Think Cache Methods
    
    async def cache_scraped_content(self, url: str, content: str, query_text: str, 
                                   request_id: str, relevance_score: float = 0.0):
        """
        Cache scraped content with deduplication using content hash
        
        Args:
            url: Source URL
            content: Scraped content text
            query_text: Query that generated this content
            request_id: Associated deep-think request ID
            relevance_score: AI-evaluated relevance score (0-10)
            
        Returns:
            Tuple of (cache_hit: bool, content_hash: str)
        """
        try:
            import hashlib
            
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Check if content already exists
            existing = await self.deepthink_cache.find_one({'content_hash': content_hash})
            
            if existing:
                # Update hit count and request tracking
                await self.deepthink_cache.update_one(
                    {'content_hash': content_hash},
                    {
                        '$inc': {'hit_count': 1},
                        '$set': {'last_accessed': datetime.utcnow()},
                        '$addToSet': {'request_ids': request_id}
                    }
                )
                logger.info(f"Content cache hit for URL: {url}")
                return (True, content_hash)
            else:
                # Cache new content
                cache_data = {
                    'content_hash': content_hash,
                    'url': url,
                    'content': content,
                    'query_text': query_text,
                    'request_id': request_id,
                    'request_ids': [request_id],
                    'relevance_score': relevance_score,
                    'hit_count': 1,
                    'created_at': datetime.utcnow(),
                    'last_accessed': datetime.utcnow(),
                    'word_count': len(content.split())
                }
                
                await self.deepthink_cache.insert_one(cache_data)
                logger.info(f"Content cached for URL: {url}")
                return (False, content_hash)
                
        except Exception as e:
            logger.error(f"Error caching scraped content for {url}: {e}")
            return (False, "")
    
    async def get_cached_scraped_content(self, content_hash: str):
        """
        Retrieve cached content by hash with hit tracking
        
        Args:
            content_hash: Content hash to lookup
            
        Returns:
            Cached content data or None
        """
        try:
            cached = await self.deepthink_cache.find_one({'content_hash': content_hash})
            
            if cached:
                # Increment hit count
                await self.deepthink_cache.update_one(
                    {'content_hash': content_hash},
                    {
                        '$inc': {'hit_count': 1},
                        '$set': {'last_accessed': datetime.utcnow()}
                    }
                )
                return cached
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached content {content_hash}: {e}")
            return None
    
    async def search_cached_scraped_content(self, query_text: str, limit: int = 20):
        """
        Search cached content by query text similarity
        
        Args:
            query_text: Text to search for
            limit: Maximum results to return
            
        Returns:
            List of matching cached content
        """
        try:
            # Try MongoDB text search first
            try:
                cursor = self.deepthink_cache.find(
                    {'$text': {'$search': query_text}},
                    {'score': {'$meta': 'textScore'}}
                ).sort([('score', {'$meta': 'textScore'}), ('hit_count', -1)]).limit(limit)
                
                results = []
                async for doc in cursor:
                    results.append(doc)
                
                if results:
                    logger.info(f"Found {len(results)} cached content items via text search for query: {query_text[:50]}...")
                    return results
            except Exception as text_search_error:
                logger.debug(f"Text search failed, falling back to regex search: {text_search_error}")
            
            # Fallback to regex search if text search fails
            regex_pattern = {'$regex': query_text, '$options': 'i'}
            query = {
                '$or': [
                    {'content': regex_pattern},
                    {'query_text': regex_pattern},
                    {'url': regex_pattern}
                ]
            }
            
            cursor = self.deepthink_cache.find(query).sort('hit_count', -1).limit(limit)
            results = []
            async for doc in cursor:
                results.append(doc)
            
            logger.info(f"Found {len(results)} cached content items via regex search for query: {query_text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching cached scraped content: {e}")
            return []
    
    async def get_deepthink_cache_stats(self):
        """
        Get deep-think cache statistics and metrics
        
        Returns:
            Dictionary with cache performance metrics
        """
        try:
            total_entries = await self.deepthink_cache.count_documents({})
            
            # Calculate total hits and unique requests
            pipeline_hits = [
                {'$group': {
                    '_id': None, 
                    'total_hits': {'$sum': '$hit_count'},
                    'unique_requests': {'$sum': {'$size': '$request_ids'}},
                    'avg_relevance': {'$avg': '$relevance_score'},
                    'avg_word_count': {'$avg': '$word_count'}
                }}
            ]
            
            hits_result = []
            async for doc in self.deepthink_cache.aggregate(pipeline_hits):
                hits_result.append(doc)
            
            stats = hits_result[0] if hits_result else {}
            
            # Get most accessed content
            most_accessed = []
            cursor = self.deepthink_cache.find(
                {}, {'url': 1, 'hit_count': 1, 'relevance_score': 1}
            ).sort('hit_count', -1).limit(5)
            
            async for doc in cursor:
                most_accessed.append({
                    'url': doc.get('url', 'Unknown'),
                    'hit_count': doc.get('hit_count', 0),
                    'relevance_score': doc.get('relevance_score', 0.0)
                })
            
            return {
                'total_entries': total_entries,
                'total_hits': stats.get('total_hits', 0),
                'unique_requests': stats.get('unique_requests', 0),
                'hit_rate': round((stats.get('total_hits', 0) / max(stats.get('unique_requests', 1), 1)), 2),
                'avg_relevance_score': round(stats.get('avg_relevance', 0.0), 2),
                'avg_word_count': round(stats.get('avg_word_count', 0.0), 0),
                'most_accessed': most_accessed
            }
            
        except Exception as e:
            logger.error(f"Error getting deep-think cache stats: {e}")
            return {
                'total_entries': 0,
                'total_hits': 0,
                'unique_requests': 0,
                'hit_rate': 0,
                'avg_relevance_score': 0.0,
                'avg_word_count': 0,
                'most_accessed': []
            }
    
    # Deep-Think Results Methods
    
    async def store_deepthink_result(self, result_data):
        """
        Store complete deep-think result
        
        Args:
            result_data: Complete deep-think result data
            
        Returns:
            Inserted result ID
        """
        try:
            result_data['created_at'] = datetime.utcnow()
            result_data['updated_at'] = datetime.utcnow()
            
            result = await self.deepthink_results.insert_one(result_data)
            logger.info(f"Deep-think result stored: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            logger.error(f"Error storing deep-think result: {e}")
            raise
    
    async def get_deepthink_result(self, request_id: str):
        """
        Get deep-think result by request ID
        
        Args:
            request_id: Unique request identifier
            
        Returns:
            Deep-think result data or None
        """
        try:
            result = await self.deepthink_results.find_one({'request_id': request_id})
            return result
        except Exception as e:
            logger.error(f"Error getting deep-think result {request_id}: {e}")
            return None
    
    async def get_user_deepthink_results(self, user_id: str, limit: int = 10):
        """
        Get recent deep-think results for a user
        
        Args:
            user_id: User identifier
            limit: Maximum results to return
            
        Returns:
            List of deep-think results
        """
        try:
            cursor = self.deepthink_results.find({'user_id': user_id}) \
                .sort('created_at', -1) \
                .limit(limit)
            
            results = []
            async for doc in cursor:
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting user deep-think results: {e}")
            return []
    
    # Deep-think chat history integration methods
    
    async def get_chat_messages_with_deepthink(self, chat_id: str, limit: int = 100, skip: int = 0):
        """
        Get chat messages with enhanced deep-think data
        
        Args:
            chat_id: Chat identifier
            limit: Maximum messages to return
            skip: Number of messages to skip
            
        Returns:
            List of messages with deep-think metadata expanded
        """
        try:
            # Get regular messages
            messages = await self.get_chat_messages(chat_id, limit, skip)
            
            # Enhance messages with deep-think data
            enhanced_messages = []
            for message in messages:
                # Check if this message has deep-think data
                if message.get('deepthink_data'):
                    # Add deep-think metadata to message
                    deepthink_data = message['deepthink_data']
                    message['has_deepthink'] = True
                    message['deepthink_summary'] = {
                        'confidence_score': deepthink_data.get('confidence_score', 0.0),
                        'total_sources': deepthink_data.get('total_sources', 0),
                        'processing_time': deepthink_data.get('processing_time', 0.0),
                        'cache_hit_rate': (
                            deepthink_data.get('cache_hits', 0) / 
                            max(deepthink_data.get('cache_hits', 0) + deepthink_data.get('cache_misses', 0), 1)
                        )
                    }
                else:
                    message['has_deepthink'] = False
                
                enhanced_messages.append(message)
            
            return enhanced_messages
            
        except Exception as e:
            logger.error(f"Error getting enhanced chat messages: {e}")
            # Fallback to regular messages
            return await self.get_chat_messages(chat_id, limit, skip)
    
    async def get_deepthink_messages_for_chat(self, chat_id: str):
        """
        Get only deep-think messages from a chat
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            List of deep-think messages with full metadata
        """
        try:
            cursor = self.messages.find({
                "chat_id": chat_id,
                "type": "assistant",
                "deepthink_data": {"$exists": True}
            }).sort("timestamp", -1)
            
            deepthink_messages = []
            async for message in cursor:
                deepthink_messages.append(message)
            
            logger.info(f"Found {len(deepthink_messages)} deep-think messages for chat {chat_id}")
            return deepthink_messages
            
        except Exception as e:
            logger.error(f"Error getting deep-think messages: {e}")
            return []
    
    async def get_user_deepthink_chat_summary(self, user_id: str, days: int = 30):
        """
        Get summary of user's deep-think usage over specified period
        
        Args:
            user_id: User identifier  
            days: Number of days to look back
            
        Returns:
            Usage summary with statistics
        """
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Aggregate deep-think usage statistics
            pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "type": "assistant",
                        "deepthink_data": {"$exists": True},
                        "timestamp": {"$gte": since_date}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_deepthink_messages": {"$sum": 1},
                        "avg_confidence": {"$avg": "$deepthink_data.confidence_score"},
                        "avg_sources": {"$avg": "$deepthink_data.total_sources"},
                        "avg_processing_time": {"$avg": "$deepthink_data.processing_time"},
                        "total_cache_hits": {"$sum": "$deepthink_data.cache_hits"},
                        "total_cache_misses": {"$sum": "$deepthink_data.cache_misses"}
                    }
                }
            ]
            
            async for result in self.messages.aggregate(pipeline):
                summary = {
                    'user_id': user_id,
                    'period_days': days,
                    'total_deepthink_requests': result.get('total_deepthink_messages', 0),
                    'average_confidence': round(result.get('avg_confidence', 0.0), 3),
                    'average_sources_per_request': round(result.get('avg_sources', 0.0), 1),
                    'average_processing_time': round(result.get('avg_processing_time', 0.0), 2),
                    'total_cache_hits': result.get('total_cache_hits', 0),
                    'total_cache_misses': result.get('total_cache_misses', 0),
                    'cache_hit_rate': (
                        result.get('total_cache_hits', 0) / 
                        max(result.get('total_cache_hits', 0) + result.get('total_cache_misses', 0), 1)
                    )
                }
                return summary
            
            # No results found
            return {
                'user_id': user_id,
                'period_days': days,
                'total_deepthink_requests': 0,
                'average_confidence': 0.0,
                'average_sources_per_request': 0.0,
                'average_processing_time': 0.0,
                'total_cache_hits': 0,
                'total_cache_misses': 0,
                'cache_hit_rate': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting deep-think chat summary: {e}")
            return None
    
    async def search_deepthink_content(self, query: str, user_id: str = None, limit: int = 10):
        """
        Search across deep-think results for relevant content
        
        Args:
            query: Search query
            user_id: Optional user filter
            limit: Maximum results to return
            
        Returns:
            List of matching deep-think results
        """
        try:
            # Create search filter
            match_filter = {
                "type": "assistant",
                "deepthink_data": {"$exists": True},
                "$or": [
                    {"message": {"$regex": query, "$options": "i"}},
                    {"deepthink_data.question": {"$regex": query, "$options": "i"}},
                    {"deepthink_data.comprehensive_answer": {"$regex": query, "$options": "i"}}
                ]
            }
            
            # Add user filter if specified
            if user_id:
                match_filter["user_id"] = user_id
            
            cursor = self.messages.find(match_filter) \
                .sort("timestamp", -1) \
                .limit(limit)
            
            search_results = []
            async for message in cursor:
                # Extract relevant info for search results
                deepthink_data = message.get('deepthink_data', {})
                search_result = {
                    'message_id': str(message['_id']),
                    'chat_id': message['chat_id'],
                    'question': deepthink_data.get('question', ''),
                    'summary_answer': deepthink_data.get('summary_answer', message.get('message', '')),
                    'confidence_score': deepthink_data.get('confidence_score', 0.0),
                    'total_sources': deepthink_data.get('total_sources', 0),
                    'timestamp': message['timestamp'],
                    'relevance': self._calculate_search_relevance(query, message)
                }
                search_results.append(search_result)
            
            # Sort by relevance (highest first)
            search_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            logger.info(f"Found {len(search_results)} deep-think results for query: '{query}'")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching deep-think content: {e}")
            return []
    
    def _calculate_search_relevance(self, query: str, message: dict) -> float:
        """
        Calculate relevance score for search results
        
        Args:
            query: Search query
            message: Message document
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        try:
            query_lower = query.lower()
            score = 0.0
            
            # Check question relevance (40% weight)
            deepthink_data = message.get('deepthink_data', {})
            question = deepthink_data.get('question', '').lower()
            if query_lower in question:
                score += 0.4
            elif any(word in question for word in query_lower.split()):
                score += 0.2
            
            # Check answer relevance (40% weight)
            answer = message.get('message', '').lower()
            if query_lower in answer:
                score += 0.4
            elif any(word in answer for word in query_lower.split()):
                score += 0.2
            
            # Boost score based on confidence (20% weight)
            confidence = deepthink_data.get('confidence_score', 0.0)
            score += confidence * 0.2
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating search relevance: {e}")
            return 0.0
