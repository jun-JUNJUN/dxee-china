from flask import Flask
from flask_cors import CORS
from meilisearch_python_sdk import Client

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Initialize Meilisearch client
    app.meilisearch = Client('http://localhost:7701', 'masterKey')
    
    # Register blueprints
    from .routes import main, search, chat
    app.register_blueprint(main.bp)
    app.register_blueprint(search.bp)
    app.register_blueprint(chat.bp)
    
    return app 
