#!/usr/bin/env python3
"""
Enhanced Chat Storage - Improved chat history storage for Deep Think results

Provides enhanced chat message storage with Deep Think session integration,
rich metadata, search optimization, and result formatting.

Key Features:
- Enhanced Deep Think result storage
- Session metadata integration
- Rich content formatting and indexing
- Search optimization for results
- Result versioning and updates
- Statistics and analytics storage
"""

import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .mongodb_service import MongoDBService
from .session_manager_service import SessionManagerService
from ..models.session_models import ProcessingSession, SessionResult

logger = logging.getLogger(__name__)

# =============================================================================
# Enhanced Chat Models
# =============================================================================

@dataclass
class DeepThinkChatResult:
    """Enhanced Deep Think result for chat storage"""
    message_id: str
    chat_id: str
    user_id: str
    session_id: str
    original_question: str
    formatted_answer: str
    raw_answer_content: str
    confidence: float
    sources: List[str]
    statistics: Dict[str, Any]
    processing_metrics: Dict[str, Any]
    session_metadata: Dict[str, Any]
    timestamp: datetime
    search_keywords: List[str] = field(default_factory=list)
    result_version: int = 1
    
    def to_chat_message_doc(self) -> Dict[str, Any]:
        """Convert to MongoDB chat message document"""
        return {
            'message_id': self.message_id,
            'chat_id': self.chat_id,
            'user_id': self.user_id,
            'message': self.formatted_answer,
            'timestamp': self.timestamp,
            'type': 'assistant',
            'search_results': self.sources,
            'shared': False,
            
            # Enhanced Deep Think metadata
            'deepthink_data': {
                'session_id': self.session_id,
                'original_question': self.original_question,
                'raw_answer': self.raw_answer_content,
                'confidence': self.confidence,
                'statistics': self.statistics,
                'processing_metrics': self.processing_metrics,
                'session_metadata': self.session_metadata,
                'result_version': self.result_version
            },
            'deepthink_completed': True,
            'deepthink_session_id': self.session_id,  # For easy querying
            
            # Search optimization
            'search_keywords': self.search_keywords,
            'content_length': len(self.formatted_answer),
            'confidence_level': self._get_confidence_level(),
            'source_count': len(self.sources),
            'has_statistics': len(self.statistics.get('numbers_found', [])) > 0,
            
            # Timestamps for analytics
            'created_at': self.timestamp,
            'updated_at': self.timestamp
        }
    
    def _get_confidence_level(self) -> str:
        """Get confidence level category"""
        if self.confidence >= 0.8:
            return 'high'
        elif self.confidence >= 0.6:
            return 'medium'
        else:
            return 'low'

# =============================================================================
# Enhanced Chat Storage Service
# =============================================================================

class EnhancedChatStorageService:
    """
    Enhanced chat storage service with Deep Think integration
    
    Provides improved storage, retrieval, and indexing of Deep Think results
    in chat history with rich metadata and search capabilities.
    """
    
    def __init__(self, mongodb_service: MongoDBService):
        self.mongodb = mongodb_service
        self._initialized = False
        
        # Storage metrics
        self.messages_stored = 0
        self.deepthink_results_stored = 0
        self.search_indexes_created = 0
        
        logger.info("EnhancedChatStorageService initialized")
    
    async def initialize(self):
        """Initialize enhanced chat storage with indexes"""
        if not self._initialized:
            try:
                await self._create_enhanced_indexes()
                self._initialized = True
                logger.info("Enhanced chat storage initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced chat storage: {e}")
                raise
    
    async def _create_enhanced_indexes(self):
        """Create enhanced indexes for Deep Think results"""
        try:
            # Standard chat indexes
            await self.mongodb.db.messages.create_index("chat_id")
            await self.mongodb.db.messages.create_index("user_id")
            await self.mongodb.db.messages.create_index("timestamp")
            await self.mongodb.db.messages.create_index("type")
            
            # Enhanced Deep Think indexes
            await self.mongodb.db.messages.create_index("deepthink_session_id")
            await self.mongodb.db.messages.create_index("deepthink_completed")
            await self.mongodb.db.messages.create_index([
                ("deepthink_completed", 1),
                ("timestamp", -1)
            ])
            
            # Search optimization indexes
            await self.mongodb.db.messages.create_index("search_keywords")
            await self.mongodb.db.messages.create_index("confidence_level")
            await self.mongodb.db.messages.create_index("content_length")
            await self.mongodb.db.messages.create_index([
                ("user_id", 1),
                ("deepthink_completed", 1),
                ("timestamp", -1)
            ])
            
            # Text search index for content
            await self.mongodb.db.messages.create_index([
                ("message", "text"),
                ("deepthink_data.original_question", "text"),
                ("search_keywords", "text")
            ], name="content_text_search")
            
            self.search_indexes_created += 7
            logger.info("Enhanced chat storage indexes created")
            
        except Exception as e:
            logger.error(f"Failed to create enhanced indexes: {e}")
            raise
    
    # =============================================================================
    # Enhanced Storage Methods
    # =============================================================================
    
    async def store_deepthink_result(self, session: ProcessingSession, 
                                   session_result: SessionResult) -> str:
        """Store Deep Think result with enhanced metadata"""
        try:
            # Generate message ID
            message_id = str(uuid.uuid4())
            
            # Format answer for display
            formatted_answer = self._format_deepthink_answer(session, session_result)
            
            # Extract search keywords from question and answer
            search_keywords = self._extract_search_keywords(
                session.question, 
                session_result.answer_content
            )
            
            # Create enhanced result object
            chat_result = DeepThinkChatResult(
                message_id=message_id,
                chat_id=session.chat_id,
                user_id=session.user_id,
                session_id=session.session_id,
                original_question=session.question,
                formatted_answer=formatted_answer,
                raw_answer_content=session_result.answer_content,
                confidence=session_result.confidence,
                sources=session_result.sources,
                statistics=session_result.statistics,
                processing_metrics={
                    'total_duration': session_result.total_duration,
                    'generation_time': session_result.generation_time,
                    'queries_generated': session_result.queries_generated,
                    'sources_analyzed': session_result.sources_analyzed,
                    'cache_hits': session_result.cache_hits
                },
                session_metadata=session.session_metadata,
                timestamp=datetime.now(timezone.utc),
                search_keywords=search_keywords
            )
            
            # Store in MongoDB
            chat_doc = chat_result.to_chat_message_doc()
            await self.mongodb.db.messages.insert_one(chat_doc)
            
            # Update metrics
            self.messages_stored += 1
            self.deepthink_results_stored += 1
            
            logger.info(f"Enhanced Deep Think result stored: {message_id} for session {session.session_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to store enhanced Deep Think result: {e}")
            raise
    
    async def update_deepthink_result(self, message_id: str, 
                                    updated_content: str, 
                                    metadata_updates: Dict[str, Any] = None) -> bool:
        """Update existing Deep Think result"""
        try:
            update_doc = {
                'message': updated_content,
                'updated_at': datetime.now(timezone.utc),
                'deepthink_data.result_version': {'$inc': 1}
            }
            
            if metadata_updates:
                for key, value in metadata_updates.items():
                    update_doc[f'deepthink_data.{key}'] = value
            
            result = await self.mongodb.db.messages.update_one(
                {'message_id': message_id, 'deepthink_completed': True},
                {'$set': update_doc}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated Deep Think result: {message_id}")
                return True
            else:
                logger.warning(f"No Deep Think result found to update: {message_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update Deep Think result {message_id}: {e}")
            return False
    
    # =============================================================================
    # Enhanced Retrieval Methods  
    # =============================================================================
    
    async def get_deepthink_results(self, user_id: str, limit: int = 20, 
                                  confidence_filter: str = None) -> List[Dict[str, Any]]:
        """Get Deep Think results with filtering options"""
        try:
            query = {
                'user_id': user_id,
                'deepthink_completed': True
            }
            
            # Add confidence filter
            if confidence_filter in ['high', 'medium', 'low']:
                query['confidence_level'] = confidence_filter
            
            cursor = self.mongodb.db.messages.find(query).sort("timestamp", -1).limit(limit)
            results = []
            
            async for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            logger.info(f"Retrieved {len(results)} Deep Think results for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get Deep Think results for user {user_id}: {e}")
            return []
    
    async def search_deepthink_results(self, user_id: str, search_query: str, 
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """Search Deep Think results by content"""
        try:
            # Text search query
            query = {
                'user_id': user_id,
                'deepthink_completed': True,
                '$text': {'$search': search_query}
            }
            
            cursor = self.mongodb.db.messages.find(
                query,
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit)
            
            results = []
            async for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            logger.info(f"Found {len(results)} Deep Think results for search: {search_query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search Deep Think results: {e}")
            return []
    
    async def get_chat_with_deepthink_context(self, chat_id: str, user_id: str) -> Dict[str, Any]:
        """Get chat messages with Deep Think context"""
        try:
            # Get all messages for chat
            messages = await self.mongodb.get_messages_by_chat_id(chat_id, user_id)
            
            # Enhance with Deep Think metadata
            enhanced_messages = []
            deepthink_count = 0
            
            for message in messages:
                if message.get('deepthink_completed'):
                    deepthink_count += 1
                    # Add summary metadata
                    deepthink_data = message.get('deepthink_data', {})
                    message['deepthink_summary'] = {
                        'confidence': deepthink_data.get('confidence', 0),
                        'sources_count': len(deepthink_data.get('sources', [])),
                        'has_statistics': len(deepthink_data.get('statistics', {}).get('numbers_found', [])) > 0,
                        'processing_time': deepthink_data.get('processing_metrics', {}).get('total_duration', 0)
                    }
                
                enhanced_messages.append(message)
            
            return {
                'chat_id': chat_id,
                'messages': enhanced_messages,
                'total_messages': len(enhanced_messages),
                'deepthink_results': deepthink_count,
                'has_deepthink': deepthink_count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get enhanced chat {chat_id}: {e}")
            return {'chat_id': chat_id, 'messages': [], 'error': str(e)}
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    def _format_deepthink_answer(self, session: ProcessingSession, 
                                result: SessionResult) -> str:
        """Format Deep Think answer for display"""
        confidence = result.confidence
        confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
        
        formatted_parts = [
            f"**Deep Think Research Result** {confidence_emoji}",
            "",
            "## 📋 Research Answer",
            result.answer_content,
            "",
            f"**Confidence:** {confidence:.1%}",
        ]
        
        # Add statistics if available
        if result.statistics:
            stats = result.statistics
            stats_parts = []
            
            if stats.get('numbers_found'):
                stats_parts.append(f"**Numbers found:** {len(stats['numbers_found'])}")
            if stats.get('percentages'):
                stats_parts.append(f"**Percentages:** {len(stats['percentages'])}")
            if stats.get('years'):
                stats_parts.append(f"**Years:** {len(stats['years'])}")
            if stats.get('currencies'):
                stats_parts.append(f"**Currencies:** {len(stats['currencies'])}")
            
            if stats_parts:
                formatted_parts.extend([
                    "",
                    "## 📊 Research Statistics",
                    *[f"- {part}" for part in stats_parts],
                    f"- **Sources analyzed:** {result.sources_analyzed}",
                    f"- **Cache hits:** {result.cache_hits}",
                ])
        
        # Add top sources
        if result.sources:
            formatted_parts.extend([
                "",
                "## 🔗 Top Sources",
            ])
            for i, source in enumerate(result.sources[:3], 1):
                formatted_parts.append(f"{i}. {source}")
        
        # Add processing info
        formatted_parts.append(f"\n*Processing time: {result.total_duration:.1f}s*")
        
        return "\n".join(formatted_parts)
    
    def _extract_search_keywords(self, question: str, answer: str) -> List[str]:
        """Extract search keywords from question and answer"""
        import re
        
        # Combine question and answer
        text = f"{question} {answer}"
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
            'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'that', 'with',
            'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been',
            'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just',
            'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
            'well', 'were', 'what'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) >= 4]
        
        # Return unique keywords, limited to top 20
        return list(set(keywords))[:20]
    
    # =============================================================================
    # Analytics and Statistics
    # =============================================================================
    
    async def get_deepthink_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get Deep Think usage analytics for user"""
        try:
            pipeline = [
                {'$match': {'user_id': user_id, 'deepthink_completed': True}},
                {'$group': {
                    '_id': None,
                    'total_results': {'$sum': 1},
                    'avg_confidence': {'$avg': '$deepthink_data.confidence'},
                    'avg_processing_time': {'$avg': '$deepthink_data.processing_metrics.total_duration'},
                    'total_sources': {'$sum': '$source_count'},
                    'high_confidence_count': {
                        '$sum': {'$cond': [{'$eq': ['$confidence_level', 'high']}, 1, 0]}
                    },
                    'medium_confidence_count': {
                        '$sum': {'$cond': [{'$eq': ['$confidence_level', 'medium']}, 1, 0]}
                    },
                    'low_confidence_count': {
                        '$sum': {'$cond': [{'$eq': ['$confidence_level', 'low']}, 1, 0]}
                    }
                }}
            ]
            
            result = await self.mongodb.db.messages.aggregate(pipeline).to_list(length=1)
            
            if result:
                analytics = result[0]
                analytics['user_id'] = user_id
                analytics.pop('_id', None)
                return analytics
            else:
                return {'user_id': user_id, 'total_results': 0}
                
        except Exception as e:
            logger.error(f"Failed to get Deep Think analytics for {user_id}: {e}")
            return {'user_id': user_id, 'error': str(e)}
    
    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage service metrics"""
        return {
            'total_messages_stored': self.messages_stored,
            'deepthink_results_stored': self.deepthink_results_stored,
            'search_indexes_created': self.search_indexes_created,
            'service_initialized': self._initialized
        }