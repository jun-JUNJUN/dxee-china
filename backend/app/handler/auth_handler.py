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
import smtplib
from email.message import EmailMessage
import random

# Get logger
logger = logging.getLogger(__name__)

async def send_verification_email(email, code):
    """
    Sends a verification code to the user's email address.
    """
    try:
        smtp_host = os.environ.get('SMTP_HOST')
        smtp_port = int(os.environ.get('SMTP_PORT', 587))
        smtp_user = os.environ.get('SMTP_USER')
        smtp_password = os.environ.get('SMTP_PASSWORD')
        sender_email = os.environ.get('SENDER_EMAIL', smtp_user)

        if not all([smtp_host, smtp_port, smtp_user, smtp_password, sender_email]):
            logger.error("SMTP settings are not fully configured in the environment.")
            # In a real app, you might have a fallback or a more robust notification system.
            # For this example, we'll log the error and the code.
            logger.error(f"VERIFICATION CODE for {email}: {code}")
            return True # Pretend it was sent

        msg = EmailMessage()
        msg.set_content(f'Your verification code is: {code}')
        msg['Subject'] = 'Your DeepSchina Verification Code'
        msg['From'] = sender_email
        msg['To'] = email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Verification email sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send verification email to {email}: {e}")
        logger.error(traceback.format_exc())
        return False

async def send_password_reset_email(email, reset_token, base_url="http://localhost:8888"):
    """
    Sends a password reset email to the user's email address.
    """
    try:
        smtp_host = os.environ.get('SMTP_HOST')
        smtp_port = int(os.environ.get('SMTP_PORT', 587))
        smtp_user = os.environ.get('SMTP_USER')
        smtp_password = os.environ.get('SMTP_PASSWORD')
        sender_email = os.environ.get('SENDER_EMAIL', smtp_user)

        if not all([smtp_host, smtp_port, smtp_user, smtp_password, sender_email]):
            logger.error("SMTP settings are not fully configured in the environment.")
            # In a real app, you might have a fallback or a more robust notification system.
            # For this example, we'll log the error and the reset link.
            reset_url = f"{base_url}/auth/reset-password?token={reset_token}"
            logger.error(f"PASSWORD RESET LINK for {email}: {reset_url}")
            return True # Pretend it was sent

        # Create reset URL
        reset_url = f"{base_url}/auth/reset-password?token={reset_token}"
        
        # HTML email content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Password Reset - DeepSchina</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #ff9776;">Password Reset Request</h2>
        <p>Hello,</p>
        <p>You requested a password reset for your DeepSchina account. Click the button below to reset your password:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_url}"
               style="background-color: #ff9776; color: white; padding: 12px 30px;
                      text-decoration: none; border-radius: 5px; display: inline-block;">
                Reset Password
            </a>
        </div>
        
        <p>Or copy and paste this link into your browser:</p>
        <p style="word-break: break-all; background-color: #f5f5f5; padding: 10px; border-radius: 3px;">
            {reset_url}
        </p>
        
        <p><strong>This link will expire in 30 minutes.</strong></p>
        
        <p>If you didn't request this password reset, please ignore this email. Your password will remain unchanged.</p>
        
        <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
        <p style="font-size: 12px; color: #666;">
            This is an automated message from DeepSchina. Please do not reply to this email.
        </p>
    </div>
</body>
</html>
        """
        
        # Plain text content
        text_content = f"""
Password Reset Request - DeepSchina

Hello,

You requested a password reset for your DeepSchina account.

Please visit the following link to reset your password:
{reset_url}

This link will expire in 30 minutes.

If you didn't request this password reset, please ignore this email. Your password will remain unchanged.

---
This is an automated message from DeepSchina. Please do not reply to this email.
        """

        msg = EmailMessage()
        msg.set_content(text_content)
        msg.add_alternative(html_content, subtype='html')
        msg['Subject'] = 'Password Reset - DeepSchina'
        msg['From'] = sender_email
        msg['To'] = email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Password reset email sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send password reset email to {email}: {e}")
        logger.error(traceback.format_exc())
        return False

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
            
            # Check if user already exists and is verified
            existing_user = await self.application.mongodb.get_user_by_email_and_auth_type(email, 'email')
            if existing_user and existing_user.get('is_verified'):
                logger.warning(f"Verified user with email {email} already exists")
                self.set_status(409)
                self.write({'error': 'A verified user with this email already exists'})
                return
            
            # Generate verification code
            verification_code = str(random.randint(100000, 999999))
            verification_expires = datetime.utcnow() + timedelta(minutes=15) # 15-minute expiry

            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            if existing_user: # User exists but is not verified, update them
                user_id = existing_user['_id']
                await self.application.mongodb.update_user(user_id, {
                    'password': hashed_password.decode('utf-8'),
                    'verification_code': verification_code,
                    'verification_expires': verification_expires,
                    'is_verified': False,
                })
                logger.info(f"Updated unverified user {email} with new verification code.")
            else: # Create a new unverified user
                user_doc = {
                    'email': email,  # Will be masked and hashed in create_user
                    'username': username,
                    'password': hashed_password.decode('utf-8'),
                    'auth_type': 'email',
                    'is_verified': False,
                    'verification_code': verification_code,
                    'verification_expires': verification_expires
                }
                user_id = await self.application.mongodb.create_user(user_doc)
                logger.info(f"Created unverified user {email} with ID: {user_id}")

            # Send verification email
            email_sent = await send_verification_email(email, verification_code)
            
            if email_sent:
                self.write({
                    'success': True,
                    'message': 'Registration successful. Please check your email for a verification code.',
                    'email': email
                })
            else:
                self.set_status(500)
                self.write({'error': 'Failed to send verification email.'})

        except Exception as e:
            logger.error(f"Unexpected error in RegisterHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class EmailVerificationHandler(BaseAuthHandler):
    """
    Handler for verifying user's email address
    """
    async def post(self):
        try:
            data = json.loads(self.request.body)
            email = data.get('email')
            code = data.get('code')

            if not email or not code:
                self.set_status(400)
                self.write({'error': 'Email and verification code are required'})
                return

            user = await self.application.mongodb.get_user_by_email_and_auth_type(email, 'email')

            if not user:
                self.set_status(404)
                self.write({'error': 'User not found'})
                return

            if user.get('is_verified'):
                self.set_status(400)
                self.write({'error': 'Email is already verified'})
                return

            if user.get('verification_code') != code:
                self.set_status(400)
                self.write({'error': 'Invalid verification code'})
                return
            
            if datetime.utcnow() > user.get('verification_expires'):
                self.set_status(400)
                self.write({'error': 'Verification code has expired'})
                return
            
            # Verification successful
            await self.application.mongodb.update_user(user['_id'], {
                'is_verified': True,
                'verification_code': None, # Clear the code
                'verification_expires': None,
                'email_verified_at': datetime.utcnow()
            })
            
            user_id = str(user['_id'])
            self.set_current_user(user_id)
            token = self.generate_jwt_token(user_id)

            self.write({
                'success': True,
                'message': 'Email verified successfully.',
                'user_id': user_id,
                'token': token
            })

        except Exception as e:
            logger.error(f"Unexpected error in EmailVerificationHandler: {e}")
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
            
            # Get user from MongoDB with email auth
            user = await self.application.mongodb.get_user_by_email_and_auth_type(email, 'email')
            if not user:
                logger.warning(f"User with email {email} and email auth not found")
                self.set_status(401)
                self.write({'error': 'Invalid email or password'})
                return
            
            # Check if user is verified
            if not user.get('is_verified'):
                logger.warning(f"Login attempt from unverified user {email}")
                self.set_status(401)
                # To prevent user enumeration, we give a generic error.
                # But we can also send a specific one to the client to prompt for verification.
                self.write({'error': 'Please verify your email before logging in.', 'email_not_verified': True, 'email': email})
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
    async def get(self):
        try:
            # Clear the secure cookie
            self.set_current_user(None)
            
            # Also clear any other cookies that might be set
            self.clear_cookie("user_id")
            
            # Clear all cookies by setting them to expire
            for cookie_name in self.request.cookies:
                self.clear_cookie(cookie_name)
            
            logger.info("User logged out successfully")
            self.write({'success': True, 'message': 'Logged out successfully'})
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            self.write({'success': False, 'error': 'Logout failed'})

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
                
                # Check if user exists with Google auth
                email = user_info.get('email')
                existing_user = await self.application.mongodb.get_user_by_email_and_auth_type(email, 'google')
                
                if existing_user:
                    # Update existing Google user
                    user_id = str(existing_user.get('_id'))
                    await self.application.mongodb.update_user(user_id, {
                        'last_login': datetime.utcnow(),
                        'google_id': user_info.get('id')
                    })
                else:
                    # Create new Google user
                    user_doc = {
                        'email': email,  # Will be masked and hashed in create_user
                        'username': user_info.get('name'),
                        'google_id': user_info.get('id'),
                        'auth_type': 'google',
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
            
            # Check if user exists with Google auth
            email = user_info.get('email')
            existing_user = await self.application.mongodb.get_user_by_email_and_auth_type(email, 'google')
            
            if existing_user:
                # Update existing Google user
                user_id = str(existing_user.get('_id'))
                await self.application.mongodb.update_user(user_id, {
                    'last_login': datetime.utcnow(),
                    'google_id': user_info.get('sub')
                })
                # Get updated user data
                user_data = await self.application.mongodb.get_user_by_id(user_id)
            else:
                # Create new Google user
                user_doc = {
                    'email': email,  # Will be masked and hashed in create_user
                    'username': user_info.get('name'),
                    'google_id': user_info.get('sub'),
                    'auth_type': 'google',
                    'last_login': datetime.utcnow()
                }
                user_id = await self.application.mongodb.create_user(user_doc)
                user_id = str(user_id)
                user_data = await self.application.mongodb.get_user_by_id(user_id)
            
            # Set the current user
            self.set_current_user(user_id)
            
            # Generate JWT token
            token = self.generate_jwt_token(user_id)
            
            # Return success response
            self.write({
                'success': True,
                'user_id': user_id,
                'username': user_data.get('username'),
                'email_masked': user_data.get('email_masked'),
                'token': token,
                'user': {
                    'id': user_id,
                    'username': user_data.get('username'),
                    'email_masked': user_data.get('email_masked'),
                    'user_uid': user_data.get('user_uid'),
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
                
                # Check if user exists with Microsoft auth
                email = user_info.get('mail') or user_info.get('userPrincipalName')
                existing_user = await self.application.mongodb.get_user_by_email_and_auth_type(email, 'microsoft')
                
                if existing_user:
                    # Update existing Microsoft user
                    user_id = str(existing_user.get('_id'))
                    await self.application.mongodb.update_user(user_id, {
                        'last_login': datetime.utcnow(),
                        'microsoft_id': user_info.get('id')
                    })
                else:
                    # Create new Microsoft user
                    user_doc = {
                        'email': email,  # Will be masked and hashed in create_user
                        'username': user_info.get('displayName'),
                        'microsoft_id': user_info.get('id'),
                        'auth_type': 'microsoft',
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
                
                # Check if user exists with GitHub auth
                existing_user = await self.application.mongodb.get_user_by_email_and_auth_type(primary_email, 'github')
                
                if existing_user:
                    # Update existing GitHub user
                    user_id = str(existing_user.get('_id'))
                    await self.application.mongodb.update_user(user_id, {
                        'last_login': datetime.utcnow(),
                        'github_id': user_info.get('id'),
                        'github_username': user_info.get('login')
                    })
                else:
                    # Create new GitHub user
                    user_doc = {
                        'email': primary_email,  # Will be masked and hashed in create_user
                        'username': user_info.get('login') or user_info.get('name') or primary_email.split('@')[0],
                        'github_id': user_info.get('id'),
                        'github_username': user_info.get('login'),
                        'auth_type': 'github',
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
            
            # Check if user exists with GitHub auth
            existing_user = await self.application.mongodb.get_user_by_email_and_auth_type(primary_email, 'github')
            
            if existing_user:
                # Update existing GitHub user
                user_id = str(existing_user.get('_id'))
                await self.application.mongodb.update_user(user_id, {
                    'last_login': datetime.utcnow(),
                    'github_id': user_info.get('id'),
                    'github_username': user_info.get('login')
                })
                # Get updated user data
                user_data = await self.application.mongodb.get_user_by_id(user_id)
            else:
                # Create new GitHub user
                user_doc = {
                    'email': primary_email,  # Will be masked and hashed in create_user
                    'username': user_info.get('login') or user_info.get('name') or primary_email.split('@')[0],
                    'github_id': user_info.get('id'),
                    'github_username': user_info.get('login'),
                    'auth_type': 'github',
                    'last_login': datetime.utcnow()
                }
                user_id = await self.application.mongodb.create_user(user_doc)
                user_id = str(user_id)
                user_data = await self.application.mongodb.get_user_by_id(user_id)
            
            # Set the current user
            self.set_current_user(user_id)
            
            # Generate JWT token
            token = self.generate_jwt_token(user_id)
            
            # Return success response
            self.write({
                'success': True,
                'user_id': user_id,
                'username': user_data.get('username'),
                'email_masked': user_data.get('email_masked'),
                'token': token,
                'user': {
                    'id': user_id,
                    'username': user_data.get('username'),
                    'email_masked': user_data.get('email_masked'),
                    'user_uid': user_data.get('user_uid'),
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

class ForgotPasswordHandler(BaseAuthHandler):
    """
    Handler for forgot password requests
    """
    async def post(self):
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                email = data.get('email')
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            if not email:
                logger.warning("Email is required for password reset")
                self.set_status(400)
                self.write({'error': 'Email is required'})
                return
            
            # Check rate limiting first
            rate_limit_ok = await self.application.mongodb.check_reset_rate_limit(email, 'email')
            if not rate_limit_ok:
                logger.warning(f"Rate limit exceeded for password reset: {email}")
                self.set_status(429)
                self.write({
                    'error': 'Too many reset requests. Please wait before trying again.',
                    'retry_after': 3600
                })
                return
            
            # Check if user exists with email auth (don't reveal if user doesn't exist)
            user = await self.application.mongodb.get_user_by_email_and_auth_type(email, 'email')
            
            # Always return success to prevent user enumeration
            # But only send email if user actually exists
            if user and user.get('is_verified'):
                # Generate secure reset token
                import secrets
                reset_token = secrets.token_urlsafe(24)[:32]
                
                # Set token expiration (30 minutes from now)
                from datetime import datetime, timedelta
                reset_token_expires = datetime.utcnow() + timedelta(minutes=30)
                
                # Store reset token in database
                token_stored = await self.application.mongodb.update_user_reset_token(
                    email, 'email', reset_token, reset_token_expires
                )
                
                if token_stored:
                    # Send password reset email
                    base_url = self.request.protocol + "://" + self.request.host
                    email_sent = await send_password_reset_email(email, reset_token, base_url)
                    
                    if email_sent:
                        logger.info(f"Password reset email sent to {email}")
                    else:
                        logger.error(f"Failed to send password reset email to {email}")
                else:
                    logger.error(f"Failed to store reset token for {email}")
            else:
                # User doesn't exist or isn't verified, but don't reveal this
                logger.info(f"Password reset requested for non-existent or unverified user: {email}")
            
            # Always return success message to prevent user enumeration
            self.write({
                'success': True,
                'message': 'If an account with this email exists, you will receive a password reset link.'
            })
            
        except Exception as e:
            logger.error(f"Unexpected error in ForgotPasswordHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})

class ResetPasswordHandler(BaseAuthHandler):
    """
    Handler for password reset confirmation
    """
    async def get(self):
        """
        Display password reset form (validate token)
        """
        try:
            reset_token = self.get_argument('token', None)
            
            if not reset_token:
                logger.warning("Reset token is required")
                self.set_status(400)
                self.render('reset_password.html', error="Invalid reset link")
                return
            
            # Validate reset token
            user = await self.application.mongodb.get_user_by_reset_token(reset_token)
            
            if not user:
                logger.warning(f"Invalid or expired reset token: {reset_token[:8]}...")
                self.set_status(400)
                self.render('reset_password.html', error="Invalid or expired reset link")
                return
            
            # Token is valid, show reset form
            self.render('reset_password.html',
                       email_masked=user.get('email_masked'),
                       reset_token=reset_token,
                       error=None)
            
        except Exception as e:
            logger.error(f"Unexpected error in ResetPasswordHandler GET: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.render('reset_password.html', error="Server error occurred")
    
    async def post(self):
        """
        Process password reset
        """
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                reset_token = data.get('token')
                new_password = data.get('new_password')
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return
            
            if not reset_token or not new_password:
                logger.warning("Reset token and new password are required")
                self.set_status(400)
                self.write({'error': 'Reset token and new password are required'})
                return
            
            # Validate password strength
            if len(new_password) < 8:
                self.set_status(400)
                self.write({'error': 'Password must be at least 8 characters long'})
                return
            
            # Validate reset token
            user = await self.application.mongodb.get_user_by_reset_token(reset_token)
            
            if not user:
                logger.warning(f"Invalid or expired reset token: {reset_token[:8]}...")
                self.set_status(400)
                self.write({'error': 'Invalid or expired reset token'})
                return
            
            # Hash the new password
            import bcrypt
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            
            # Update user password
            user_id = str(user['_id'])
            update_result = await self.application.mongodb.update_user(user_id, {
                'password': hashed_password.decode('utf-8')
            })
            
            if update_result > 0:
                # Clear the reset token
                await self.application.mongodb.clear_reset_token(user_id)
                
                logger.info(f"Password reset successful for user: {user_id}")
                self.write({
                    'success': True,
                    'message': 'Password has been reset successfully.'
                })
            else:
                logger.error(f"Failed to update password for user: {user_id}")
                self.set_status(500)
                self.write({'error': 'Failed to update password'})
            
        except Exception as e:
            logger.error(f"Unexpected error in ResetPasswordHandler POST: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})
