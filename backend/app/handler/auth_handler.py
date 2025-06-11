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
            self.set_secure_cookie("user_id", str(user_id), expires_days=60)
        else:
            self.clear_cookie("user_id")
    
    def generate_jwt_token(self, user_id, expiration_days=60):
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
                
                # Redirect to the main page with auth success parameter
                self.redirect('/?auth=success')
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
    
    async def post(self):
        """
        Handle Google Identity Services (One Tap) authentication
        """
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                id_token = data.get('id_token')
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            if not id_token:
                logger.warning("ID token is required")
                self.set_status(400)
                self.write({'error': 'ID token is required'})
                return
            
            # Verify the ID token with Google
            import requests
            verify_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            response = requests.get(verify_url)
            
            if response.status_code != 200:
                logger.error(f"Failed to verify Google ID token: {response.text}")
                self.set_status(401)
                self.write({'error': 'Invalid Google ID token'})
                return
            
            user_info = response.json()
            
            # Verify the audience (client ID)
            if user_info.get('aud') != self.settings['google_oauth']['client_id']:
                logger.error("Invalid audience in Google ID token")
                self.set_status(401)
                self.write({'error': 'Invalid token audience'})
                return
            
            # Check if user exists
            email = user_info.get('email')
            existing_user = await self.application.mongodb.get_user_by_email(email)
            
            if existing_user:
                # Update existing user
                user_id = str(existing_user.get('_id'))
                await self.application.mongodb.update_user(user_id, {
                    'last_login': datetime.utcnow(),
                    'google_id': user_info.get('sub')
                })
                user_data = existing_user
            else:
                # Create new user
                user_doc = {
                    'email': email,
                    'username': user_info.get('name'),
                    'google_id': user_info.get('sub'),
                    'auth_type': 'google',
                    'created_at': datetime.utcnow(),
                    'last_login': datetime.utcnow()
                }
                user_id = await self.application.mongodb.create_user(user_doc)
                user_id = str(user_id)
                user_data = user_doc
                user_data['_id'] = user_id
            
            # Set the current user
            self.set_current_user(user_id)
            
            # Generate JWT token
            token = self.generate_jwt_token(user_id)
            
            # Return success response
            self.write({
                'success': True,
                'user_id': user_id,
                'username': user_data.get('username'),
                'email': user_data.get('email'),
                'token': token,
                'user': {
                    'id': user_id,
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'auth_type': 'google'
                }
            })
        except Exception as e:
            logger.error(f"Unexpected error in Google OAuth POST handler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class SessionCheckHandler(BaseAuthHandler):
    """
    Handler to check if user is logged in
    """
    async def get(self):
        try:
            user_id = self.get_current_user()
            if user_id:
                # Get user from MongoDB
                user = await self.application.mongodb.get_user_by_id(user_id)
                if user:
                    # Remove sensitive information
                    if 'password' in user:
                        del user['password']
                    
                    # Convert MongoDB document to JSON-serializable format
                    user['_id'] = str(user['_id'])
                    
                    # Convert all datetime fields to ISO format strings
                    for key, value in user.items():
                        if isinstance(value, datetime):
                            user[key] = value.isoformat()
                    
                    self.write({
                        'logged_in': True,
                        'user': user
                    })
                else:
                    # User ID in cookie but user not found in DB
                    self.set_current_user(None)
                    self.write({'logged_in': False})
            else:
                self.write({'logged_in': False})
        except Exception as e:
            logger.error(f"Error in SessionCheckHandler: {e}")
            logger.error(traceback.format_exc())
            self.write({'logged_in': False})

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

class GitHubOAuthHandler(BaseAuthHandler, OAuth2Mixin):
    """
    Handler for GitHub OAuth authentication
    """
    _OAUTH_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
    _OAUTH_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
    
    async def get(self):
        if self.get_argument('code', False):
            # This is the OAuth callback
            try:
                # Exchange the code for an access token
                import tornado.httpclient
                import urllib.parse
                
                body = {
                    'client_id': self.settings['github_oauth']['client_id'],
                    'client_secret': self.settings['github_oauth']['client_secret'],
                    'code': self.get_argument('code'),
                    'redirect_uri': self.settings['github_oauth']['redirect_uri']
                }
                
                http_client = tornado.httpclient.AsyncHTTPClient()
                response = await http_client.fetch(
                    self._OAUTH_ACCESS_TOKEN_URL,
                    method="POST",
                    headers={'Accept': 'application/json'},
                    body=urllib.parse.urlencode(body)
                )
                
                access_data = json.loads(response.body)
                access_token = access_data.get('access_token')
                
                if not access_token:
                    logger.error(f"Failed to get access token from GitHub: {access_data}")
                    self.set_status(401)
                    self.write({'error': 'Failed to authenticate with GitHub'})
                    return
                
                # Get user info from GitHub
                user_response = await http_client.fetch(
                    "https://api.github.com/user",
                    headers={
                        'Authorization': f"Bearer {access_token}",
                        'Accept': 'application/vnd.github.v3+json',
                        'User-Agent': 'Dxee-Chat-App'
                    }
                )
                
                user_info = json.loads(user_response.body)
                
                # Get user email from GitHub (emails might be private)
                email_response = await http_client.fetch(
                    "https://api.github.com/user/emails",
                    headers={
                        'Authorization': f"Bearer {access_token}",
                        'Accept': 'application/vnd.github.v3+json',
                        'User-Agent': 'Dxee-Chat-App'
                    }
                )
                
                emails = json.loads(email_response.body)
                primary_email = None
                for email_data in emails:
                    if email_data.get('primary', False):
                        primary_email = email_data.get('email')
                        break
                
                if not primary_email and emails:
                    primary_email = emails[0].get('email')
                
                if not primary_email:
                    logger.error("Could not get email from GitHub user")
                    self.set_status(401)
                    self.write({'error': 'Could not get email from GitHub account'})
                    return
                
                # Check if user exists
                existing_user = await self.application.mongodb.get_user_by_email(primary_email)
                
                if existing_user:
                    # Update existing user
                    user_id = str(existing_user.get('_id'))
                    await self.application.mongodb.update_user(user_id, {
                        'last_login': datetime.utcnow(),
                        'github_id': user_info.get('id'),
                        'github_username': user_info.get('login')
                    })
                else:
                    # Create new user
                    user_doc = {
                        'email': primary_email,
                        'username': user_info.get('login') or user_info.get('name') or primary_email.split('@')[0],
                        'github_id': user_info.get('id'),
                        'github_username': user_info.get('login'),
                        'auth_type': 'github',
                        'created_at': datetime.utcnow(),
                        'last_login': datetime.utcnow()
                    }
                    user_id = await self.application.mongodb.create_user(user_doc)
                    user_id = str(user_id)
                
                # Set the current user
                self.set_current_user(user_id)
                
                # Redirect to the main page with auth success parameter
                self.redirect('/?auth=success')
            except Exception as e:
                logger.error(f"Error in GitHub OAuth callback: {e}")
                logger.error(traceback.format_exc())
                self.set_status(500)
                self.write({'error': f'Error authenticating with GitHub: {str(e)}'})
        else:
            # Start the OAuth flow
            self.authorize_redirect(
                redirect_uri=self.settings['github_oauth']['redirect_uri'],
                client_id=self.settings['github_oauth']['client_id'],
                scope=['user:email'],
                response_type='code',
                extra_params={'allow_signup': 'true'}
            )
    
    async def post(self):
        """
        Handle GitHub OAuth for AJAX requests (similar to Google's POST handler)
        """
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                code = data.get('code')
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            if not code:
                logger.warning("Authorization code is required")
                self.set_status(400)
                self.write({'error': 'Authorization code is required'})
                return
            
            # Exchange code for access token
            import tornado.httpclient
            import urllib.parse
            
            body = {
                'client_id': self.settings['github_oauth']['client_id'],
                'client_secret': self.settings['github_oauth']['client_secret'],
                'code': code,
                'redirect_uri': self.settings['github_oauth']['redirect_uri']
            }
            
            http_client = tornado.httpclient.AsyncHTTPClient()
            response = await http_client.fetch(
                self._OAUTH_ACCESS_TOKEN_URL,
                method="POST",
                headers={'Accept': 'application/json'},
                body=urllib.parse.urlencode(body)
            )
            
            access_data = json.loads(response.body)
            access_token = access_data.get('access_token')
            
            if not access_token:
                logger.error(f"Failed to get access token from GitHub: {access_data}")
                self.set_status(401)
                self.write({'error': 'Failed to authenticate with GitHub'})
                return
            
            # Get user info from GitHub
            user_response = await http_client.fetch(
                "https://api.github.com/user",
                headers={
                    'Authorization': f"Bearer {access_token}",
                    'Accept': 'application/vnd.github.v3+json',
                    'User-Agent': 'Dxee-Chat-App'
                }
            )
            
            user_info = json.loads(user_response.body)
            
            # Get user email from GitHub
            email_response = await http_client.fetch(
                "https://api.github.com/user/emails",
                headers={
                    'Authorization': f"Bearer {access_token}",
                    'Accept': 'application/vnd.github.v3+json',
                    'User-Agent': 'Dxee-Chat-App'
                }
            )
            
            emails = json.loads(email_response.body)
            primary_email = None
            for email_data in emails:
                if email_data.get('primary', False):
                    primary_email = email_data.get('email')
                    break
            
            if not primary_email and emails:
                primary_email = emails[0].get('email')
            
            if not primary_email:
                logger.error("Could not get email from GitHub user")
                self.set_status(401)
                self.write({'error': 'Could not get email from GitHub account'})
                return
            
            # Check if user exists
            existing_user = await self.application.mongodb.get_user_by_email(primary_email)
            
            if existing_user:
                # Update existing user
                user_id = str(existing_user.get('_id'))
                await self.application.mongodb.update_user(user_id, {
                    'last_login': datetime.utcnow(),
                    'github_id': user_info.get('id'),
                    'github_username': user_info.get('login')
                })
                user_data = existing_user
            else:
                # Create new user
                user_doc = {
                    'email': primary_email,
                    'username': user_info.get('login') or user_info.get('name') or primary_email.split('@')[0],
                    'github_id': user_info.get('id'),
                    'github_username': user_info.get('login'),
                    'auth_type': 'github',
                    'created_at': datetime.utcnow(),
                    'last_login': datetime.utcnow()
                }
                user_id = await self.application.mongodb.create_user(user_doc)
                user_id = str(user_id)
                user_data = user_doc
                user_data['_id'] = user_id
            
            # Set the current user
            self.set_current_user(user_id)
            
            # Generate JWT token
            token = self.generate_jwt_token(user_id)
            
            # Return success response
            self.write({
                'success': True,
                'user_id': user_id,
                'username': user_data.get('username'),
                'email': user_data.get('email'),
                'token': token,
                'user': {
                    'id': user_id,
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'auth_type': 'github'
                }
            })
        except Exception as e:
            logger.error(f"Unexpected error in GitHub OAuth POST handler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

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
