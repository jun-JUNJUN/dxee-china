import os
import tornado.web
import tornado.ioloop
import asyncio
import logging
import traceback
from dotenv import load_dotenv
from meilisearch_python_sdk import AsyncClient
from app.handler.search_handler import SearchHandler
from app.handler.health_handler import HealthHandler
from app.handler.chat_handler import ChatMessageHandler, ChatHistoryHandler, UserChatsHandler, ShareMessageHandler, SharedMessagesHandler, ChatStreamHandler
from app.handler.main_handler import MainHandler, NotFoundHandler, FaviconHandler
from app.handler.auth_handler import RegisterHandler, LoginHandler, LogoutHandler, GoogleOAuthHandler, GitHubOAuthHandler, MicrosoftOAuthHandler, AppleOAuthHandler, UserProfileHandler, SessionCheckHandler
from app.handler.deep_search_handler import DeepSearchHandler, DeepSearchStreamHandler, DeepSearchWebSocketHandler
from app.handler.dual_research_handler import DualResearchHandler, DualResearchWebSocketHandler
from app.service.deepseek_service import DeepSeekService
from app.service.mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='backend.log',
    filemode='a'  # Append mode
)
# Add console handler to see logs in the console as well
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)
logger.info("Logging configured to write to backend.log")

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded from .env file")

# Global variables
input_queue = None
output_queue = None
deepseek_service = None

class Application(tornado.web.Application):
    def __init__(self, input_queue=None, output_queue=None):
        handlers = [
            (r"/search", SearchHandler),
            (r"/health", HealthHandler),
            (r"/chat/message", ChatMessageHandler),
            (r"/chat/stream", ChatStreamHandler),
            (r"/chat/history/([^/]+)", ChatHistoryHandler),
            (r"/chat/user", UserChatsHandler),
            (r"/chat/share/([^/]+)", ShareMessageHandler),
            (r"/chat/shared", SharedMessagesHandler),
            (r"/deep-search", DeepSearchHandler),
            (r"/deep-search/stream", DeepSearchStreamHandler),
            (r"/deep-search/ws", DeepSearchWebSocketHandler),
            (r"/dual-research", DualResearchHandler),
            (r"/dual-research/ws", DualResearchWebSocketHandler),
            (r"/auth/register", RegisterHandler),
            (r"/auth/login", LoginHandler),
            (r"/auth/logout", LogoutHandler),
            (r"/auth/google", GoogleOAuthHandler),
            (r"/auth/google/callback", GoogleOAuthHandler),
            (r"/auth/github", GitHubOAuthHandler),
            (r"/auth/github/callback", GitHubOAuthHandler),
            (r"/auth/microsoft", MicrosoftOAuthHandler),
            (r"/auth/apple", AppleOAuthHandler),
            (r"/auth/profile", UserProfileHandler),
            (r"/auth/session", SessionCheckHandler),
            (r"/favicon.ico", FaviconHandler),  # Favicon handler
            (r"/", MainHandler),  # Main page handler
        ]
        
        settings = {
            'debug': True,
            'autoreload': True,
            'template_path': 'templates',  # Path to templates directory
            'default_handler_class': NotFoundHandler,  # 404 handler
            'cookie_secret': os.environ.get('AUTH_SECRET_KEY', 'default_secret_key_change_in_production'),
            'login_url': '/auth/login',
            'google_oauth': {
                'client_id': os.environ.get('GOOGLE_CLIENT_ID', ''),
                'client_secret': os.environ.get('GOOGLE_CLIENT_SECRET', ''),
                'redirect_uri': os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:8888/auth/google')
            },
            'github_oauth': {
                'client_id': os.environ.get('GITHUB_CLIENT_ID', ''),
                'client_secret': os.environ.get('GITHUB_CLIENT_SECRET', ''),
                'redirect_uri': os.environ.get('GITHUB_REDIRECT_URI', 'http://localhost:8100/auth/github/callback')
            },
            'microsoft_oauth': {
                'client_id': os.environ.get('MICROSOFT_CLIENT_ID', ''),
                'client_secret': os.environ.get('MICROSOFT_CLIENT_SECRET', ''),
                'redirect_uri': os.environ.get('MICROSOFT_REDIRECT_URI', 'http://localhost:8888/auth/microsoft')
            },
            'apple_oauth': {
                'client_id': os.environ.get('APPLE_CLIENT_ID', ''),
                'client_secret': os.environ.get('APPLE_CLIENT_SECRET', ''),
                'redirect_uri': os.environ.get('APPLE_REDIRECT_URI', 'http://localhost:8888/auth/apple')
            }
        }
        
        super().__init__(handlers, **settings)
        
        # Initialize Meilisearch client from environment variables
        meilisearch_host = os.environ.get('MEILISEARCH_HOST', 'http://localhost:7701')
        meilisearch_api_key = os.environ.get('MEILISEARCH_API_KEY', 'masterKey')
        logger.info(f"Initializing Meilisearch client with host: {meilisearch_host}")
        self.meilisearch = AsyncClient(meilisearch_host, meilisearch_api_key)
        
        # Initialize MongoDB service
        io_loop = tornado.ioloop.IOLoop.current()
        self.mongodb = MongoDBService(io_loop=io_loop)
        logger.info("MongoDB service initialized")
        
        # Store queue references
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Initialize stream queues for streaming responses
        self.stream_queues = {}

def make_app():
    """
    Factory function that creates and returns the Tornado application.
    This is called by Gunicorn.
    """
    global input_queue, output_queue, deepseek_service

    # Initialize queues if they haven't been initialized
    if input_queue is None and output_queue is None:
        logger.info("Initializing input and output queues")
        # Initialize queues
        input_queue = []
        output_queue = []
        
        # Initialize DeepSeek service
        logger.info("Initializing DeepSeek service")
        deepseek_service = DeepSeekService(input_queue, output_queue)

    # Create and return the Tornado application
    # MongoDB service will be initialized within the Application class
    return Application(input_queue, output_queue)

# This is the application callable that Gunicorn expects
app = make_app()

# Start the DeepSeek service processing loop
async def start_deepseek_service(app_instance=None):
    global deepseek_service
    if deepseek_service:
        logger.info("Starting DeepSeek service processing loop")
        try:
            # Pass stream_queues if available from app instance
            stream_queues = getattr(app_instance, 'stream_queues', None) if app_instance else None
            await deepseek_service.start_processing(stream_queues)
        except Exception as e:
            logger.error(f"Error in DeepSeek service processing loop: {e}")
            logger.error(traceback.format_exc())

# Create MongoDB indexes
async def create_mongodb_indexes(app_instance):
    try:
        if hasattr(app_instance, 'mongodb'):
            logger.info("Creating MongoDB indexes")
            await app_instance.mongodb.create_indexes()
            logger.info("MongoDB indexes created successfully")
        else:
            logger.warning("MongoDB service not found in application instance")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")
        logger.error(traceback.format_exc())

# Add the background tasks to the IOLoop
def start_background_tasks(app_instance=None):
    logger.info("Starting background tasks")
    asyncio.ensure_future(start_deepseek_service(app_instance))
    if app_instance:
        asyncio.ensure_future(create_mongodb_indexes(app_instance))

if __name__ == "__main__":
    # This is for running the application directly (not through Gunicorn)
    port = int(os.environ.get('PORT', 8888))
    app.listen(port)
    logger.info(f"Server started at http://localhost:{port}")

    # Start background tasks
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.add_callback(lambda: start_background_tasks(app))

    try:
        logger.info("Starting IOLoop")
        io_loop.start()
    except KeyboardInterrupt:
        logger.info("Stopping server due to keyboard interrupt")
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        logger.error(traceback.format_exc())
