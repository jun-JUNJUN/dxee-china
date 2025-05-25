class SearchService:
    def __init__(self, meilisearch_client):
        self.meilisearch = meilisearch_client
    
    async def search(self, query, context=None):
        """
        Perform a search with optional context
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
            search_params['filter'] = f'context_id = {context}'
        
        # Perform the search
        results = await self.meilisearch.index('documents').search(
            query,
            search_params
        )
        
        return {
            'results': results['hits'],
            'estimatedTotalHits': results['estimatedTotalHits'],
            'processingTimeMs': results['processingTimeMs']
        }
    
    async def index_document(self, document):
        """
        Index a new document
        """
        return await self.meilisearch.index('documents').add_documents([document])
    
    async def update_document(self, document):
        """
        Update an existing document
        """
        return await self.meilisearch.index('documents').update_documents([document])
    
    async def delete_document(self, document_id):
        """
        Delete a document
        """
        return await self.meilisearch.index('documents').delete_document(document_id) 
