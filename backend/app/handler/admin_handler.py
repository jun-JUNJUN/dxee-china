import tornado.web
import json
import logging
import os
from datetime import datetime
from bson import json_util
from .auth_handler import BaseAuthHandler

logger = logging.getLogger(__name__)

class AdminHandler(BaseAuthHandler):
    """
    Handler for admin dashboard
    """
    @property
    def ADMIN_USERS(self):
        """Get admin users from environment variable (as email hashes for secure comparison)"""
        admin_email = os.environ.get('ADMIN_EMAIL', 'admin@dxee.work')
        # Return the email hash for secure comparison with user.email_hash
        admin_email_hash = self.application.mongodb._generate_email_hash(admin_email, 'email')
        return {admin_email_hash}
    
    async def is_admin(self):
        """Check if current user is an admin"""
        user_id = self.get_current_user()
        logger.info(f"Admin check - user_id: {user_id}")
        if not user_id:
            logger.info("Admin check - no user_id found")
            return False
        try:
            user = await self.application.mongodb.get_user_by_id(user_id)
            logger.info(f"Admin check - user found: {user is not None}")
            if not user:
                logger.info("Admin check - user not found in database")
                return False
            
            user_email_hash = user.get('email_hash')
            admin_email_hashes = self.ADMIN_USERS
            logger.info(f"Admin check - user email hash: {user_email_hash}")
            logger.info(f"Admin check - admin email hashes: {admin_email_hashes}")
            
            is_admin = user_email_hash in admin_email_hashes
            logger.info(f"Admin check - is admin: {is_admin}")
            return is_admin
        except Exception as e:
            logger.error(f"Error checking admin status: {e}")
            return False

    @tornado.web.authenticated
    async def head(self):
        """Check if current user has admin access (for admin status check)"""
        if not await self.is_admin():
            self.set_status(403)
            return
        self.set_status(200)

    @tornado.web.authenticated
    async def get(self):
        """Render admin dashboard"""
        if not await self.is_admin():
            self.set_status(403)
            return self.write({'error': 'Unauthorized'})

        try:
            # Get all users
            users_cursor = self.application.mongodb.users.find({})
            users = []
            async for user in users_cursor:
                # Remove sensitive data
                if 'password' in user:
                    del user['password']
                users.append(user)

            # Get all chats
            chats_cursor = self.application.mongodb.chats.find({})
            chats = []
            async for chat in chats_cursor:
                chats.append(chat)

            # Get recent messages
            messages_cursor = self.application.mongodb.messages.find({}).sort('timestamp', -1).limit(100)
            messages = []
            async for message in messages_cursor:
                messages.append(message)

            # Convert MongoDB documents to JSON-serializable format
            for doc in users + chats + messages:
                doc['_id'] = str(doc['_id'])
                for key, value in doc.items():
                    if isinstance(value, datetime):
                        doc[key] = value.isoformat()

            # Render template
            self.render(
                "admin.html",
                users=users,
                chats=chats,
                messages=messages
            )

        except Exception as e:
            logger.error(f"Error in admin dashboard: {e}")
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'}) 
