import os
import requests
import logging
import traceback
from flask import Blueprint, request, jsonify, current_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get backend URL from environment variables
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8888')

bp = Blueprint('chat', __name__, url_prefix='/chat')

@bp.route('/message', methods=['POST'])
def send_message():
    data = request.get_json()
    message = data.get('message')
    chat_id = data.get('chat_id')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        logger.info(f"Sending message to backend: {message[:30]}... with chat_id: {chat_id}")
        
        # Forward the request to the backend
        backend_url = f'{BACKEND_URL}/chat/message'
        try:
            response = requests.post(
                backend_url,
                json={
                    'message': message,
                    'chat_id': chat_id
                },
                timeout=30  # Increase timeout for longer requests
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                error_msg = f'Backend error ({response.status_code}): {response.text}'
                logger.error(error_msg)
                return jsonify({'error': error_msg}), response.status_code
            
            # Log successful response
            result = response.json()
            logger.info(f"Received response from backend for chat_id: {result.get('chat_id')}")
            
            # Return the response from the backend
            return jsonify(result)
        except requests.exceptions.Timeout:
            error_msg = "Backend request timed out. The DeepSeek API may be taking too long to respond."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 504
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to backend server. Please check if the backend is running."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 503
    except Exception as e:
        # Log the full exception with traceback
        logger.error(f"Error in send_message: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@bp.route('/history/<chat_id>', methods=['GET'])
def get_history(chat_id):
    try:
        logger.info(f"Getting chat history for chat_id: {chat_id}")
        
        # Forward the request to the backend
        backend_url = f'{BACKEND_URL}/chat/history/{chat_id}'
        try:
            response = requests.get(backend_url, timeout=10)
            
            # Check if the request was successful
            if response.status_code != 200:
                error_msg = f'Backend error ({response.status_code}): {response.text}'
                logger.error(error_msg)
                return jsonify({'error': error_msg}), response.status_code
            
            # Log successful response
            result = response.json()
            message_count = len(result.get('messages', []))
            logger.info(f"Retrieved {message_count} messages for chat_id: {chat_id}")
            
            # Return the response from the backend
            return jsonify(result)
        except requests.exceptions.Timeout:
            error_msg = "Backend request timed out when retrieving chat history."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 504
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to backend server. Please check if the backend is running."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 503
    except Exception as e:
        # Log the full exception with traceback
        logger.error(f"Error in get_history: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
