import os
import logging
import motor.motor_tornado
import hashlib
import uuid
from datetime import datetime
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
