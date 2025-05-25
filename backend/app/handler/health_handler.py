import tornado.web
import json

class HealthHandler(tornado.web.RequestHandler):
    async def get(self):
        self.write({'status': 'healthy'})
