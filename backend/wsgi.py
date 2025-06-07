import os
import sys
import asyncio
import logging
import tornado.ioloop

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

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger.info("Added current directory to Python path")

try:
    # Import the Tornado application and background task function
    logger.info("Importing Tornado application and background task function")
    from app.tornado_main import app, start_background_tasks

    # Initialize the IOLoop for Gunicorn workers
    logger.info("Initializing IOLoop for Gunicorn workers")
    io_loop = tornado.ioloop.IOLoop.current()

    # Add a callback to start background tasks after the server starts
    logger.info("Adding callback to start background tasks")
    io_loop.add_callback(lambda: start_background_tasks(app))

    # This is the WSGI application callable that Gunicorn expects
    application = app
    logger.info("WSGI application initialized successfully")
except Exception as e:
    logger.error(f"Error initializing WSGI application: {e}")
    logger.exception("Exception details:")
    raise
