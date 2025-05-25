import uuid
from datetime import datetime

def process_chat_message(meilisearch_client, message, chat_id=None):
    """
    Process a chat message and return relevant search results
    """
    if not chat_id:
        chat_id = str(uuid.uuid4())
    
    # Store the message in chat history
    message_doc = {
        'id': str(uuid.uuid4()),
        'chat_id': chat_id,
        'message': message,
        'timestamp': datetime.utcnow().isoformat(),
        'type': 'user'
    }
    
    meilisearch_client.index('chat_history').add_documents([message_doc])
    
    # Search for relevant documents
    search_results = meilisearch_client.index('documents').search(
        message,
        {
            'limit': 5,
            'attributesToSearchOn': ['title', 'content', 'tags'],
            'attributesToRetrieve': ['*']
        }
    )
    
    # Create response message
    response_doc = {
        'id': str(uuid.uuid4()),
        'chat_id': chat_id,
        'message': 'Here are the relevant results for your query.',
        'timestamp': datetime.utcnow().isoformat(),
        'type': 'assistant',
        'search_results': search_results['hits']
    }
    
    meilisearch_client.index('chat_history').add_documents([response_doc])
    
    return {
        'chat_id': chat_id,
        'response': response_doc
    }

def get_chat_history(meilisearch_client, chat_id):
    """
    Retrieve chat history for a specific chat
    """
    results = meilisearch_client.index('chat_history').search(
        '',
        {
            'filter': f'chat_id = {chat_id}',
            'sort': ['timestamp:asc'],
            'limit': 100
        }
    )
    
    return {
        'chat_id': chat_id,
        'messages': results['hits']
    } 
