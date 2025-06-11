import tornado.web
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainHandler(tornado.web.RequestHandler):
    """
    Handler for the main page
    """
    def get(self):
        logger.info("Serving main page")
        google_client_id = self.application.settings.get('google_oauth', {}).get('client_id', '')
        self.render("index.html", google_client_id=google_client_id)

class FaviconHandler(tornado.web.RequestHandler):
    """
    Handler for favicon requests
    """
    def get(self):
        # Return a 204 No Content response for favicon requests
        # This prevents 404 errors in the logs
        self.set_status(204)
        self.finish()

class NotFoundHandler(tornado.web.RequestHandler):
    """
    Handler for 404 errors
    """
    def prepare(self):
        logger.info("404 error - Page not found")
        self.set_status(404)
        self.render("404.html")
