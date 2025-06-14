#!/usr/bin/env python3
"""
Dual Research Handler - API endpoint for dual research approach
Runs both intelligent and legacy research methods simultaneously
"""

import tornado.web
import tornado.websocket
import tornado.gen
import json
import logging
import asyncio
from typing import Dict, Any
from ..service.dual_research_service import DualResearchService

logger = logging.getLogger(__name__)

class DualResearchHandler(tornado.web.RequestHandler):
    """Handler for Dual Research API requests"""
    
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
        """Handle Dual Research requests - runs both intelligent and legacy research"""
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
            
            logger.info(f"Dual research request: {question}")
            
            # Initialize Dual Research service
            dual_research = DualResearchService()
            
            # Run both research approaches in parallel
            logger.info("Running intelligent and legacy research in parallel")
            intelligent_task = dual_research.conduct_intelligent_research(question)
            legacy_task = dual_research.conduct_legacy_research(question)
            
            # Wait for both to complete
            intelligent_result, legacy_result = yield [intelligent_task, legacy_task]
            
            # Return both results
            self.write({
                'status': 'success',
                'intelligent_research': intelligent_result,
                'legacy_research': legacy_result,
                'question': question,
                'timestamp': intelligent_result.get('timestamp', '')
            })
            
        except Exception as e:
            logger.error(f"Dual research error: {e}")
            self.set_status(500)
            self.write({
                'status': 'error',
                'message': str(e)
            })

class DualResearchWebSocketHandler(tornado.websocket.WebSocketHandler):
    """WebSocket handler for real-time Dual Research progress updates"""
    
    def check_origin(self, origin):
        """Allow connections from any origin (configure as needed for production)"""
        return True
    
    def open(self):
        """Called when WebSocket connection is opened"""
        logger.info("Dual Research WebSocket connection opened")
        self.write_message({
            'type': 'connection',
            'status': 'connected'
        })
    
    def on_close(self):
        """Called when WebSocket connection is closed"""
        logger.info("Dual Research WebSocket connection closed")
    
    @tornado.gen.coroutine
    def on_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'start_dual_research':
                question = data.get('question', '').strip()
                
                if not question:
                    self.write_message({
                        'type': 'error',
                        'message': 'Question is required'
                    })
                    return
                
                logger.info(f"WebSocket dual research request: {question}")
                
                # Progress callback for real-time updates
                def progress_callback(step: str, step_data: Dict[str, Any]):
                    self.write_message({
                        'type': 'progress',
                        'step': step,
                        'data': step_data
                    })
                
                # Initialize Dual Research service with progress callback
                dual_research = DualResearchService(progress_callback=progress_callback)
                
                # Send start confirmation
                self.write_message({
                    'type': 'start',
                    'question': question
                })
                
                # Conduct dual research
                results = yield dual_research.conduct_dual_research(question)
                
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
            logger.error(f"WebSocket dual research error: {e}")
            self.write_message({
                'type': 'error',
                'message': str(e)
            })