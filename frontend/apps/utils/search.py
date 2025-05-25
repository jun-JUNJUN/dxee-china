def process_search_query(meilisearch_client, query, context=None):
    """
    Process a search query with optional context from chat history
    """
    search_params = {
        'q': query,
        'limit': 20,
        'offset': 0,
        'attributesToSearchOn': ['title', 'content', 'tags'],
        'attributesToRetrieve': ['*'],
        'attributesToHighlight': ['title', 'content'],
        'highlightPreTag': '<em>',
        'highlightPostTag': '</em>'
    }
    
    if context:
        # Add context-aware search parameters
        search_params['filter'] = f'context_id = {context}'
    
    # Perform the search
    results = meilisearch_client.index('documents').search(
        query,
        search_params
    )
    
    return {
        'results': results['hits'],
        'estimatedTotalHits': results['estimatedTotalHits'],
        'processingTimeMs': results['processingTimeMs']
    } 
