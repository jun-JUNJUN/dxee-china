import tornado.web
import tornado.ioloop
from meilisearch_python_sdk import AsyncClient
from app.handler.search_handler import SearchHandler
from app.handler.health_handler import HealthHandler

# Global variables
input_queue = None
output_queue = None

class Application(tornado.web.Application):
    def __init__(self, input_queue=None, output_queue=None):
        handlers = [
            (r"/search", SearchHandler),
            (r"/health", HealthHandler),
        ]
        
        settings = {
            'debug': True,
            'autoreload': True
        }
        
        super().__init__(handlers, **settings)
        
        # Initialize Meilisearch client
        self.meilisearch = AsyncClient('http://localhost:7701', 'masterKey')
        
        # Store queue references
        self.input_queue = input_queue
        self.output_queue = output_queue

def make_app():
    """
    Factory function that creates and returns the Tornado application.
    This is called by Gunicorn.
    """
    global input_queue, output_queue

    # Initialize queues if they haven't been initialized
    if input_queue is None and output_queue is None:
        # Initialize your queues here
        input_queue = []  # Replace with actual queue initialization
        output_queue = []  # Replace with actual queue initialization

    # Create and return the Tornado application
    return Application(input_queue, output_queue)

# This is the application callable that Gunicorn expects
app = make_app()

if __name__ == "__main__":
    # This is for running the application directly (not through Gunicorn)
    app.listen(8888)
    print("Server started at http://localhost:8888")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nStopping server...")
        print("Server stopped")
