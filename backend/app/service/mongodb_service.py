import os
import logging
import motor.motor_tornado
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
            # Create indexes for users collection
            await self.users.create_index("email", unique=True)
            await self.users.create_index("username")
            
            # Create indexes for chats collection
            await self.chats.create_index("user_id")
            await self.chats.create_index("updated_at")
            
            # Create indexes for messages collection
            await self.messages.create_index([("chat_id", 1), ("timestamp", 1)])
            await self.messages.create_index([("user_id", 1), ("chat_id", 1)])
            await self.messages.create_index("shared", sparse=True)
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
            raise
    
    # User methods
    
    async def create_user(self, user_data):
        """
        Create a new user
        """
        try:
            user_data["created_at"] = datetime.utcnow()
            result = await self.users.insert_one(user_data)
            logger.info(f"User created with ID: {result.inserted_id}")
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
    
    async def get_user_by_email(self, email):
        """
        Get a user by email
        """
        try:
            user = await self.users.find_one({"email": email})
            return user
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
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
