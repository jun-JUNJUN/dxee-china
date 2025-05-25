import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Tornado application
from app.tornado_main import app

# This is the WSGI application callable that Gunicorn expects
application = app
