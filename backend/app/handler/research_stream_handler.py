#!/usr/bin/env python3
"""
Research Stream Handler - New implementation using the exact algorithm
from test_deepseek_advanced_web_research4_01.py with JSON logging.

This handler replaces the existing deep-think functionality with the new
research orchestrator that matches the test algorithm exactly.
"""

import tornado.web
import json
import uuid
import logging
import traceback
import asyncio
from datetime import datetime

from app.service.research_orchestrator import ResearchOrchestrator

logger = logging.getLogger(__name__)


class ResearchStreamHandler(tornado.web.RequestHandler):
    """
    Handler for streaming research using the new research orchestrator
    """

    async def post(self):
        """Handle research request with streaming progress"""
        try:
            # Parse request body
            try:
                data = json.loads(self.request.body)
                message = data.get('message')
                chat_id = data.get('chat_id')
                search_mode = data.get('search_mode', 'research')

                logger.info(f"Received research request: {message[:50]}... for chat: {chat_id or 'new chat'}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                self.set_status(400)
                self.write({'error': f'Invalid JSON: {str(e)}'})
                return

            if not message:
                logger.warning("Message is required but was not provided")
                self.set_status(400)
                self.write({'error': 'Message is required'})
                return

            # Generate a new chat ID if not provided
            if not chat_id:
                chat_id = str(uuid.uuid4())
                logger.info(f"Generated new chat_id: {chat_id}")

            # Get user_id from authentication
            user_id = self.get_secure_cookie("user_id")
            if user_id:
                user_id = user_id.decode('utf-8')
            else:
                user_id = "default_user"

            # Set headers for Server-Sent Events
            self.set_header('Content-Type', 'text/event-stream')
            self.set_header('Cache-Control', 'no-cache')
            self.set_header('Connection', 'keep-alive')
            self.set_header('Access-Control-Allow-Origin', '*')
            self.set_header('Access-Control-Allow-Headers', 'Content-Type')

            # Send initial response with chat_id
            self.write(f"data: {json.dumps({'type': 'chat_id', 'chat_id': chat_id})}\n\n")
            await self.flush()

            # Store the user message if MongoDB is available
            if hasattr(self.application, 'mongodb') and self.application.mongodb:
                try:
                    message_doc = {
                        'message_id': str(uuid.uuid4()),
                        'chat_id': chat_id,
                        'user_id': user_id,
                        'message': message,
                        'timestamp': datetime.utcnow(),
                        'type': 'user',
                        'shared': False
                    }
                    await self.application.mongodb.create_message(message_doc)
                    logger.info(f"User message stored for chat_id: {chat_id}")
                except Exception as e:
                    logger.error(f"Error storing user message: {e}")

            # Initialize research orchestrator
            try:
                orchestrator = ResearchOrchestrator()
                logger.info(f"🔬 Starting research session for: {message}")

                # Stream research progress
                async for progress in orchestrator.research_with_streaming(message):
                    progress_data = {
                        'type': 'research_step',
                        'step': progress.step,
                        'total_steps': progress.total_steps,
                        'description': progress.description,
                        'progress': progress.progress,
                        'metadata': progress.metadata
                    }

                    # Send progress update
                    self.write(f"data: {json.dumps(progress_data)}\n\n")
                    await self.flush()

                    # Check if research is complete
                    if progress.progress >= 100.0 and 'result' in progress.metadata:
                        result = progress.metadata['result']

                        # Store the AI response if MongoDB is available
                        if hasattr(self.application, 'mongodb') and self.application.mongodb:
                            try:
                                response_doc = {
                                    'message_id': str(uuid.uuid4()),
                                    'chat_id': chat_id,
                                    'user_id': user_id,
                                    'message': result['answer'],
                                    'timestamp': datetime.utcnow(),
                                    'type': 'assistant',
                                    'research_data': {
                                        'confidence': result['confidence'],
                                        'sources': result['sources'],
                                        'statistics': result['statistics'],
                                        'metadata': result['metadata'],
                                        'duration': result['duration']
                                    },
                                    'shared': False
                                }
                                await self.application.mongodb.create_message(response_doc)
                                logger.info(f"Research result stored for chat_id: {chat_id}")
                            except Exception as e:
                                logger.error(f"Error storing research result: {e}")

                        # Send final complete message
                        complete_data = {
                            'type': 'complete',
                            'content': result['answer'],
                            'research_result': result,
                            'metadata': {
                                'confidence': result['confidence'],
                                'sources_count': len(result['sources']),
                                'duration': result['duration']
                            }
                        }

                        self.write(f"data: {json.dumps(complete_data)}\n\n")
                        await self.flush()
                        break

            except Exception as e:
                logger.error(f"❌ Research orchestrator error: {e}")
                logger.error(traceback.format_exc())

                error_data = {
                    'type': 'error',
                    'content': f'Research failed: {str(e)}',
                    'error_details': str(e)
                }

                self.write(f"data: {json.dumps(error_data)}\n\n")
                await self.flush()

        except Exception as e:
            logger.error(f"Unexpected error in ResearchStreamHandler: {e}")
            logger.error(traceback.format_exc())
            self.set_status(500)
            self.write({'error': f'Server error: {str(e)}'})


class ResearchChatHandler(tornado.web.RequestHandler):
    """
    Non-streaming research handler for compatibility
    """

    async def post(self):
        """Handle research request without streaming"""
        try:
            # Parse request body
            data = json.loads(self.request.body)
            message = data.get('message')
            chat_id = data.get('chat_id', str(uuid.uuid4()))

            if not message:
                self.set_status(400)
                self.write({'error': 'Message is required'})
                return

            logger.info(f"Non-streaming research request: {message[:50]}...")

            # Initialize research orchestrator
            orchestrator = ResearchOrchestrator()

            # Execute research
            result = await orchestrator.research_with_logging(message)

            # Return result
            self.write({
                'status': 'success',
                'chat_id': chat_id,
                'result': result
            })

        except Exception as e:
            logger.error(f"Research handler error: {e}")
            self.set_status(500)
            self.write({
                'status': 'error',
                'error': str(e)
            })