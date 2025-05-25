from flask import Blueprint, request, jsonify, current_app
from ..utils.search import process_search_query

bp = Blueprint('search', __name__, url_prefix='/search')

@bp.route('', methods=['GET'])
def search():
    query = request.args.get('q', '')
    context = request.args.get('context', '')
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    try:
        results = process_search_query(
            current_app.meilisearch,
            query,
            context
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500 
