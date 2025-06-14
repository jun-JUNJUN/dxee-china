#!/usr/bin/env python3
"""
Deep Search Handler - API endpoint for advanced web research
"""

import tornado.web
import tornado.websocket
import tornado.gen
import json
import logging
import asyncio
from typing import Dict, Any
from ..service.deep_search_service import DeepSearchService

logger = logging.getLogger(__name__)

class DeepSearchHandler(tornado.web.RequestHandler):
    """Handler for Deep Search API requests with real-time progress updates"""
    
    def set_default_headers(self):
        """Set CORS and security headers"""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With, X-XSRFToken")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Content-Type", "application/json")
        
        # Security headers
        self.set_header('X-XSS-Protection', '1; mode=block')
        self.set_header('X-Frame-Options', 'SAMEORIGIN')
        self.set_header('X-Content-Type-Options', 'nosniff')

    def options(self):
        """Handle preflight CORS requests"""
        self.set_status(204)
        self.finish()

    @tornado.gen.coroutine
    def post(self):
        """Handle Deep Search requests"""
        try:
            # Parse request body
            data = json.loads(self.request.body)
            question = data.get('question', '').strip()
            
            if not question:
                self.set_status(400)
                self.write({
                    'status': 'error',
                    'message': 'Question is required'
                })
                return
            
            logger.info(f"Deep search request: {question}")
            
            # Initialize Deep Search service
            deep_search = DeepSearchService()
            
            # Conduct deep search
            results = yield deep_search.conduct_deep_search(question)
            
            # Return results
            self.write({
                'status': 'success',
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Deep search error: {e}")
            self.set_status(500)
            self.write({
                'status': 'error',
                'message': str(e)
            })

class DeepSearchStreamHandler(tornado.web.RequestHandler):
    """Handler for Deep Search with Server-Sent Events for real-time progress"""
    
    def set_default_headers(self):
        """Set SSE headers"""
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("Connection", "keep-alive")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")

    def write_sse_data(self, event_type: str, data: Dict[str, Any]):
        """Write Server-Sent Event data"""
        event_data = json.dumps(data)
        self.write(f"event: {event_type}\n")
        self.write(f"data: {event_data}\n\n")
        self.flush()

    @tornado.gen.coroutine
    def post(self):
        """Handle streaming Deep Search requests"""
        try:
            # Parse request body
            data = json.loads(self.request.body)
            question = data.get('question', '').strip()
            
            if not question:
                self.write_sse_data('error', {
                    'message': 'Question is required'
                })
                return
            
            logger.info(f"Streaming deep search request: {question}")
            
            # Progress callback for real-time updates
            def progress_callback(step: str, step_data: Dict[str, Any]):
                self.write_sse_data('progress', {
                    'step': step,
                    'data': step_data
                })
            
            # Initialize Deep Search service with progress callback
            deep_search = DeepSearchService(progress_callback=progress_callback)
            
            # Send start event
            self.write_sse_data('start', {'question': question})
            
            # Conduct deep search
            results = yield deep_search.conduct_deep_search(question)
            
            # Send completion event
            self.write_sse_data('complete', {'results': results})
            
        except Exception as e:
            logger.error(f"Streaming deep search error: {e}")
            self.write_sse_data('error', {
                'message': str(e)
            })

class DeepSearchWebSocketHandler(tornado.websocket.WebSocketHandler):
    """WebSocket handler for real-time Deep Search progress updates"""
    
    def check_origin(self, origin):
        """Allow connections from any origin (configure as needed for production)"""
        return True
    
    def open(self):
        """Called when WebSocket connection is opened"""
        logger.info("Deep Search WebSocket connection opened")
        self.write_message({
            'type': 'connection',
            'status': 'connected'
        })
    
    def on_close(self):
        """Called when WebSocket connection is closed"""
        logger.info("Deep Search WebSocket connection closed")
    
    @tornado.gen.coroutine
    def on_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'start_search':
                question = data.get('question', '').strip()
                
                if not question:
                    self.write_message({
                        'type': 'error',
                        'message': 'Question is required'
                    })
                    return
                
                logger.info(f"WebSocket deep search request: {question}")
                
                # Progress callback for real-time updates
                def progress_callback(step: str, step_data: Dict[str, Any]):
                    self.write_message({
                        'type': 'progress',
                        'step': step,
                        'data': step_data
                    })
                
                # Initialize Deep Search service with progress callback
                deep_search = DeepSearchService(progress_callback=progress_callback)
                
                # Send start confirmation
                self.write_message({
                    'type': 'start',
                    'question': question
                })
                
                # Conduct deep search
                results = yield deep_search.conduct_deep_search(question)
                
                # Send completion
                self.write_message({
                    'type': 'complete',
                    'results': results
                })
                
            else:
                self.write_message({
                    'type': 'error',
                    'message': f'Unknown action: {action}'
                })
                
        except json.JSONDecodeError:
            self.write_message({
                'type': 'error',
                'message': 'Invalid JSON message'
            })
        except Exception as e:
            logger.error(f"WebSocket deep search error: {e}")
            self.write_message({
                'type': 'error',
                'message': str(e)
            })
