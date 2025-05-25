import tornado.web
import json
from ..service.search_service import SearchService

class SearchHandler(tornado.web.RequestHandler):
    async def get(self):
        query = self.get_argument('q', '')
        context = self.get_argument('context', None)
        
        if not query:
            self.set_status(400)
            self.write({'error': 'Query parameter is required'})
            return
        
        try:
            search_service = SearchService(self.application.meilisearch)
            results = await search_service.search(query, context)
            self.write(results)
        except Exception as e:
            self.set_status(500)
            self.write({'error': str(e)}) 
