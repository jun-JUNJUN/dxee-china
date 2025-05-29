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
from app.handler.chat_handler import ChatMessageHandler, ChatHistoryHandler
from app.handler.main_handler import MainHandler, NotFoundHandler
from app.service.deepseek_service import DeepSeekService

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
            (r"/chat/history/([^/]+)", ChatHistoryHandler),
            (r"/", MainHandler),  # Main page handler
        ]
        
        settings = {
            'debug': True,
            'autoreload': True,
            'template_path': 'templates',  # Path to templates directory
            'default_handler_class': NotFoundHandler  # 404 handler
        }
        
        super().__init__(handlers, **settings)
        
        # Initialize Meilisearch client from environment variables
        meilisearch_host = os.environ.get('MEILISEARCH_HOST', 'http://localhost:7701')
        meilisearch_api_key = os.environ.get('MEILISEARCH_API_KEY', 'masterKey')
        logger.info(f"Initializing Meilisearch client with host: {meilisearch_host}")
        self.meilisearch = AsyncClient(meilisearch_host, meilisearch_api_key)
        
        # Store queue references
        self.input_queue = input_queue
        self.output_queue = output_queue

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
    return Application(input_queue, output_queue)

# This is the application callable that Gunicorn expects
app = make_app()

# Start the DeepSeek service processing loop
async def start_deepseek_service():
    global deepseek_service
    if deepseek_service:
        logger.info("Starting DeepSeek service processing loop")
        try:
            await deepseek_service.start_processing()
        except Exception as e:
            logger.error(f"Error in DeepSeek service processing loop: {e}")
            logger.error(traceback.format_exc())

# Add the DeepSeek service task to the IOLoop
def start_background_tasks():
    logger.info("Starting background tasks")
    asyncio.ensure_future(start_deepseek_service())

if __name__ == "__main__":
    # This is for running the application directly (not through Gunicorn)
    port = int(os.environ.get('PORT', 8888))
    app.listen(port)
    logger.info(f"Server started at http://localhost:{port}")

    # Start background tasks
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.add_callback(start_background_tasks)

    try:
        logger.info("Starting IOLoop")
        io_loop.start()
    except KeyboardInterrupt:
        logger.info("Stopping server due to keyboard interrupt")
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        logger.error(traceback.format_exc())
