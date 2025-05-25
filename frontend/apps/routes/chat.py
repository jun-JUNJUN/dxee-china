from flask import Blueprint, request, jsonify, current_app
from ..utils.chat import process_chat_message, get_chat_history

bp = Blueprint('chat', __name__, url_prefix='/chat')

@bp.route('/message', methods=['POST'])
def send_message():
    data = request.get_json()
    message = data.get('message')
    chat_id = data.get('chat_id')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        response = process_chat_message(
            current_app.meilisearch,
            message,
            chat_id
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/history/<chat_id>', methods=['GET'])
def get_history(chat_id):
    try:
        history = get_chat_history(current_app.meilisearch, chat_id)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500 
