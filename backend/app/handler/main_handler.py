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
        self.render("index.html")

class NotFoundHandler(tornado.web.RequestHandler):
    """
    Handler for 404 errors
    """
    def prepare(self):
        logger.info("404 error - Page not found")
        self.set_status(404)
        self.render("404.html")
