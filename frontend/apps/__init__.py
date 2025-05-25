import os
from flask import Flask
from flask_cors import CORS
from meilisearch_python_sdk import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Initialize Meilisearch client from environment variables
    meilisearch_host = os.environ.get('MEILISEARCH_HOST', 'http://localhost:7701')
    meilisearch_api_key = os.environ.get('MEILISEARCH_API_KEY', 'masterKey')
    app.meilisearch = Client(meilisearch_host, meilisearch_api_key)
    
    # Register blueprints
    from .routes import main, search, chat
    app.register_blueprint(main.bp)
    app.register_blueprint(search.bp)
    app.register_blueprint(chat.bp)
    
    return app 
