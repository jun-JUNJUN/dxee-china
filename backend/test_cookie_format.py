#!/usr/bin/env python3
"""
Test script to understand Tornado's secure cookie format
"""
import asyncio
import os
import tornado.web
import tornado.ioloop
import tornado.httputil
import tornado.testing
from dotenv import load_dotenv
from app.service.mongodb_service import MongoDBService

class TestHandler(tornado.web.RequestHandler):
    """Test handler to examine cookie behavior"""
    
    def get_current_user(self):
        """Get the current user from the secure cookie"""
        user_id = self.get_secure_cookie("user_id")
        if user_id:
            return user_id.decode('utf-8')
        return None
    
    def set_current_user(self, user_id):
        """Set the current user in the secure cookie"""
        if user_id:
            self.set_secure_cookie("user_id", str(user_id), expires_days=60)
        else:
            self.clear_cookie("user_id")

async def test_cookie_format():
    """Test cookie format"""
    print("? Testing Cookie Format...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Create a simple Tornado application
    cookie_secret = os.environ.get('AUTH_SECRET_KEY', 'default_secret_key_change_in_production')
    
    app = tornado.web.Application([
        (r"/test", TestHandler),
    ], cookie_secret=cookie_secret)
    
    print("1. Testing Tornado's secure cookie generation...")
    
    # Create a mock request
    request = tornado.httputil.HTTPServerRequest(
        method="GET",
        uri="/test",
        headers=tornado.httputil.HTTPHeaders()
    )
    
    # Create handler instance
    handler = TestHandler(app, request)
    
    # Test setting a secure cookie
    known_user_id = "6847ebd29dc6c51d64edbcde"
    handler.set_current_user(known_user_id)
    
    # Get the cookie value that was set
    cookie_header = None
    for name, value in handler._headers.get_list():
        if name.lower() == 'set-cookie' and 'user_id=' in value:
            cookie_header = value
            break
    
    if cookie_header:
        print(f"   Generated cookie header: {cookie_header}")
        
        # Extract just the cookie value
        cookie_value = None
        for part in cookie_header.split(';'):
            if part.strip().startswith('user_id='):
                cookie_value = part.strip().split('=', 1)[1]
                break
        
        if cookie_value:
            print(f"   Cookie value: {cookie_value[:50]}...")
            
            # Test reading the cookie back
            request_with_cookie = tornado.httputil.HTTPServerRequest(
                method="GET",
                uri="/test",
                headers=tornado.httputil.HTTPHeaders({
                    'Cookie': f'user_id={cookie_value}'
                })
            )
            
            handler_with_cookie = TestHandler(app, request_with_cookie)
            retrieved_user_id = handler_with_cookie.get_current_user()
            
            print(f"   Retrieved user ID: {retrieved_user_id}")
            
            if retrieved_user_id == known_user_id:
                print("   ? Cookie round-trip successful!")
            else:
                print("   ? Cookie round-trip failed!")
        else:
            print("   ? Could not extract cookie value")
    else:
        print("   ? No cookie header found")
    
    print("\n2. Testing MongoDB lookup with retrieved user ID...")
    if 'retrieved_user_id' in locals() and retrieved_user_id:
        try:
            io_loop = tornado.ioloop.IOLoop.current()
            mongodb = MongoDBService(io_loop=io_loop)
            
            user = await mongodb.get_user_by_id(retrieved_user_id)
            
            if user:
                print(f"   ? User found in database:")
                print(f"      ID: {user['_id']}")
                print(f"      Email: {user['email']}")
            else:
                print(f"   ? User not found in database")
                
        except Exception as e:
            print(f"   ? Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cookie_format())
