import tornado.web
import json
import logging
import traceback
import bcrypt
import uuid
from datetime import datetime, timedelta
import jwt
from tornado.auth import GoogleOAuth2Mixin, OAuth2Mixin
import os

# Get logger
logger = logging.getLogger(__name__)

class BaseAuthHandler(tornado.web.RequestHandler):
    """
    Base class for authentication handlers
    """
    def get_current_user(self):
        """
        Get the current user from the secure cookie
        """
        user_id = self.get_secure_cookie("user_id")
        if user_id:
            return user_id.decode('utf-8')
        return None
    
    def set_current_user(self, user_id):
        """
        Set the current user in the secure cookie
        """
        if user_id:
            self.set_secure_cookie("user_id", str(user_id), expires_days=30)
        else:
            self.clear_cookie("user_id")
    
    def generate_jwt_token(self, user_id, expiration_days=30):
        """
        Generate a JWT token for the user
        """
        secret = self.application.settings.get('cookie_secret', 'default_secret')
        payload = {
            'user_id': str(user_id),
            'exp': datetime.utcnow() + timedelta(days=expiration_days)
        }
        token = jwt.encode(payload, secret, algorithm='HS256')
        return token
    
    def verify_jwt_token(self, token):
        """
        Verify a JWT token and return the user_id
        """
        try:
            secret = self.application.settings.get('cookie_secret', 'default_secret')
            payload = jwt.decode(token, secret, algorithms=['HS256'])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None

class RegisterHandler(BaseAuthHandler):
    """
    Handler for user registration with email and password
    """
    async def post(self):
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                email = data.get('email')
                password = data.get('password')
                username = data.get('username', email.split('@')[0] if email else None)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            if not email or not password:
                logger.warning("Email and password are required")
                self.set_status(400)
                self.write({'error': 'Email and password are required'})
                return
            
            # Check if user already exists
            existing_user = await self.application.mongodb.get_user_by_email(email)
            if existing_user:
                logger.warning(f"User with email {email} already exists")
                self.set_status(409)
                self.write({'error': 'User with this email already exists'})
                return
            
            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Create user document
            user_doc = {
                'email': email,
                'username': username,
                'password': hashed_password.decode('utf-8'),
                'auth_type': 'email',
                'created_at': datetime.utcnow()
            }
            
            # Store the user in MongoDB
            try:
                user_id = await self.application.mongodb.create_user(user_doc)
                logger.info(f"User created with ID: {user_id}")
                
                # Set the current user
                self.set_current_user(str(user_id))
                
                # Generate JWT token
                token = self.generate_jwt_token(str(user_id))
                
                # Return success response
                self.write({
                    'success': True,
                    'user_id': str(user_id),
                    'token': token
                })
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error creating user: {str(e)}'})
        except Exception as e:
            logger.error(f"Unexpected error in RegisterHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class LoginHandler(BaseAuthHandler):
    """
    Handler for user login with email and password
    """
    async def post(self):
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                email = data.get('email')
                password = data.get('password')
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            if not email or not password:
                logger.warning("Email and password are required")
                self.set_status(400)
                self.write({'error': 'Email and password are required'})
                return
            
            # Get user from MongoDB
            user = await self.application.mongodb.get_user_by_email(email)
            if not user:
                logger.warning(f"User with email {email} not found")
                self.set_status(401)
                self.write({'error': 'Invalid email or password'})
                return
            
            # Check password
            stored_password = user.get('password', '').encode('utf-8')
            if not bcrypt.checkpw(password.encode('utf-8'), stored_password):
                logger.warning(f"Invalid password for user {email}")
                self.set_status(401)
                self.write({'error': 'Invalid email or password'})
                return
            
            # Set the current user
            user_id = str(user.get('_id'))
            self.set_current_user(user_id)
            
            # Generate JWT token
            token = self.generate_jwt_token(user_id)
            
            # Return success response
            self.write({
                'success': True,
                'user_id': user_id,
                'username': user.get('username'),
                'token': token
            })
        except Exception as e:
            logger.error(f"Unexpected error in LoginHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class LogoutHandler(BaseAuthHandler):
    """
    Handler for user logout
    """
    def get(self):
        self.set_current_user(None)
        self.write({'success': True})

class GoogleOAuthHandler(BaseAuthHandler, GoogleOAuth2Mixin):
    """
    Handler for Google OAuth authentication
    """
    async def get(self):
        if self.get_argument('code', False):
            # This is the OAuth callback
            try:
                # Exchange the code for an access token
                access = await self.get_authenticated_user(
                    redirect_uri=self.settings['google_oauth']['redirect_uri'],
                    code=self.get_argument('code')
                )
                
                # Get user info from Google
                user_info = await self.oauth2_request(
                    "https://www.googleapis.com/oauth2/v1/userinfo",
                    access_token=access["access_token"]
                )
                
                # Check if user exists
                email = user_info.get('email')
                existing_user = await self.application.mongodb.get_user_by_email(email)
                
                if existing_user:
                    # Update existing user
                    user_id = str(existing_user.get('_id'))
                    await self.application.mongodb.update_user(user_id, {
                        'last_login': datetime.utcnow(),
                        'google_id': user_info.get('id')
                    })
                else:
                    # Create new user
                    user_doc = {
                        'email': email,
                        'username': user_info.get('name'),
                        'google_id': user_info.get('id'),
                        'auth_type': 'google',
                        'created_at': datetime.utcnow(),
                        'last_login': datetime.utcnow()
                    }
                    user_id = await self.application.mongodb.create_user(user_doc)
                    user_id = str(user_id)
                
                # Set the current user
                self.set_current_user(user_id)
                
                # Redirect to the main page
                self.redirect('/')
            except Exception as e:
                logger.error(f"Error in Google OAuth callback: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error authenticating with Google: {str(e)}'})
        else:
            # Start the OAuth flow
            self.authorize_redirect(
                redirect_uri=self.settings['google_oauth']['redirect_uri'],
                client_id=self.settings['google_oauth']['client_id'],
                scope=['profile', 'email'],
                response_type='code',
                extra_params={'approval_prompt': 'auto'}
            )

class MicrosoftOAuthHandler(BaseAuthHandler, OAuth2Mixin):
    """
    Handler for Microsoft OAuth authentication
    """
    _OAUTH_AUTHORIZE_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
    _OAUTH_ACCESS_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    
    async def get(self):
        if self.get_argument('code', False):
            # This is the OAuth callback
            try:
                # Exchange the code for an access token
                body = {
                    'client_id': self.settings['microsoft_oauth']['client_id'],
                    'client_secret': self.settings['microsoft_oauth']['client_secret'],
                    'code': self.get_argument('code'),
                    'redirect_uri': self.settings['microsoft_oauth']['redirect_uri'],
                    'grant_type': 'authorization_code'
                }
                
                response = await self.fetch(
                    self._OAUTH_ACCESS_TOKEN_URL,
                    method="POST",
                    body=tornado.httputil.urlencode(body)
                )
                
                access = json.loads(response.body)
                
                # Get user info from Microsoft
                user_info_response = await self.fetch(
                    "https://graph.microsoft.com/v1.0/me",
                    headers={'Authorization': f"Bearer {access['access_token']}"}
                )
                
                user_info = json.loads(user_info_response.body)
                
                # Check if user exists
                email = user_info.get('mail') or user_info.get('userPrincipalName')
                existing_user = await self.application.mongodb.get_user_by_email(email)
                
                if existing_user:
                    # Update existing user
                    user_id = str(existing_user.get('_id'))
                    await self.application.mongodb.update_user(user_id, {
                        'last_login': datetime.utcnow(),
                        'microsoft_id': user_info.get('id')
                    })
                else:
                    # Create new user
                    user_doc = {
                        'email': email,
                        'username': user_info.get('displayName'),
                        'microsoft_id': user_info.get('id'),
                        'auth_type': 'microsoft',
                        'created_at': datetime.utcnow(),
                        'last_login': datetime.utcnow()
                    }
                    user_id = await self.application.mongodb.create_user(user_doc)
                    user_id = str(user_id)
                
                # Set the current user
                self.set_current_user(user_id)
                
                # Redirect to the main page
                self.redirect('/')
            except Exception as e:
                logger.error(f"Error in Microsoft OAuth callback: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error authenticating with Microsoft: {str(e)}'})
        else:
            # Start the OAuth flow
            self.authorize_redirect(
                redirect_uri=self.settings['microsoft_oauth']['redirect_uri'],
                client_id=self.settings['microsoft_oauth']['client_id'],
                scope=['User.Read'],
                response_type='code',
                extra_params={'prompt': 'select_account'}
            )

class AppleOAuthHandler(BaseAuthHandler):
    """
    Handler for Apple OAuth authentication
    """
    async def get(self):
        # Apple OAuth implementation would go here
        # This is a placeholder as Apple OAuth requires additional setup
        # including server-to-server validation and JWT token verification
        self.write({'error': 'Apple OAuth not implemented yet'})

class UserProfileHandler(BaseAuthHandler):
    """
    Handler for user profile
    """
    @tornado.web.authenticated
    async def get(self):
        try:
            user_id = self.current_user
            
            # Get user from MongoDB
            user = await self.application.mongodb.get_user_by_id(user_id)
            if not user:
                logger.warning(f"User with ID {user_id} not found")
                self.set_status(404)
                self.write({'error': 'User not found'})
                return
            
            # Remove sensitive information
            if 'password' in user:
                del user['password']
            
            # Convert MongoDB document to JSON-serializable format
            user['_id'] = str(user['_id'])
            if isinstance(user.get('created_at'), datetime):
                user['created_at'] = user['created_at'].isoformat()
            if isinstance(user.get('last_login'), datetime):
                user['last_login'] = user['last_login'].isoformat()
            
            self.write(user)
        except Exception as e:
            logger.error(f"Unexpected error in UserProfileHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})
