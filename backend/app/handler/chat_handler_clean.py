    async def _handle_deepthink_research(self, message: str, chat_id: str, user_id: str, message_id: str, stream_queue):
        """Handle deep-think research mode with background processing that continues even if client disconnects"""
        try:
            # Validate environment configuration first
            validation_errors = []
            
            # Check Serper API key
            serper_api_key = os.environ.get('SERPER_API_KEY')
            if not serper_api_key:
                validation_errors.append("SERPER_API_KEY not configured")
            
            # Check DeepSeek API key
            deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
            if not deepseek_api_key:
                validation_errors.append("DEEPSEEK_API_KEY not configured")
            
            # Check DeepSeek API URL
            deepseek_api_url = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
            if not deepseek_api_url.startswith(('http://', 'https://')):
                validation_errors.append("DEEPSEEK_API_URL is malformed")
            
            if validation_errors:
                error_msg = f"Configuration errors: {', '.join(validation_errors)}. Please check your environment variables."
                logger.error(error_msg)
                self.write(f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n")
                await self.flush()
                return
            
            # Send configuration validation success
            self.write(f"data: {json.dumps({'type': 'deepthink_progress', 'step': 0, 'total_steps': 10, 'description': '✅ Configuration validated', 'progress': 5})}\n\n")
            await self.flush()
            
            # Initialize the deep-think orchestrator
            orchestrator = DeepThinkOrchestrator(
                deepseek_service=self.application.deepseek_service,
                mongodb_service=self.application.mongodb,
                serper_api_key=serper_api_key,
                timeout=int(os.environ.get('DEEPSEEK_RESEARCH_TIMEOUT', 600)),
                max_concurrent_searches=int(os.environ.get('MAX_CONCURRENT_RESEARCH', 3)),
                cache_expiry_days=int(os.environ.get('CACHE_EXPIRY_DAYS', 30))
            )
            
            # Create deep-think request
            request = DeepThinkRequest(
                request_id=str(uuid.uuid4()),
                question=message,
                chat_id=chat_id,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                timeout_seconds=int(os.environ.get('DEEPSEEK_RESEARCH_TIMEOUT', 600))
            )
            
            # Start deep-think as background task that continues even if client disconnects
            import asyncio
            asyncio.create_task(self._background_deepthink_process(orchestrator, request, chat_id, user_id, message_id))
            
            # Stream progress updates to client for as long as they're connected (but not final result)
            try:
                async for progress_update in orchestrator.stream_deep_think(request):
                    if progress_update.step < orchestrator.total_steps:
                        # Only stream progress updates, not final result (handled by background task)
                        progress_data = {
                            'type': 'deepthink_progress',
                            'step': progress_update.step,
                            'total_steps': progress_update.total_steps,
                            'description': progress_update.description,
                            'progress': progress_update.progress_percent,
                            'details': progress_update.details
                        }
                        self.write(f"data: {json.dumps(progress_data, cls=MongoJSONEncoder)}\n\n")
                        await self.flush()
                    else:
                        # Final step reached - background task will handle storage and completion
                        self.write(f"data: {json.dumps({'type': 'deepthink_progress', 'step': progress_update.step, 'total_steps': progress_update.total_steps, 'description': '✅ Analysis complete - results saved to chat history', 'progress': 100})}\n\n")
                        await self.flush()
                        break
                        
            except asyncio.TimeoutError:
                # Client connection timeout - background task continues
                logger.info(f"⏰ Client connection timed out for deep-think, but background processing continues for chat_id: {chat_id}")
                timeout_details = {
                    'type': 'info',
                    'content': "🕐 Deep-think analysis is taking longer than expected.\n\n"
                              "✅ **Your analysis is continuing in the background.**\n"
                              "📱 You can safely close this page - results will be saved to your chat history.\n"
                              "🔄 Refresh the page in a few minutes to see the completed analysis.",
                    'timeout_seconds': request.timeout_seconds
                }
                self.write(f"data: {json.dumps(timeout_details)}\n\n")
                await self.flush()
                
        except Exception as e:
            logger.error(f"Deep-think setup error: {e}")
            logger.error(traceback.format_exc())
            self.write(f"data: {json.dumps({'type': 'error', 'content': f'Deep-think setup failed: {str(e)}'})}\n\n")
            await self.flush()

    async def _background_deepthink_process(self, orchestrator, request, chat_id: str, user_id: str, message_id: str):
        """
        Background process that ensures deep-think completes and saves to MongoDB 
        even if client disconnects
        """
        try:
            logger.info(f"🔄 Starting background deep-think process for chat_id: {chat_id}")
            
            # Process the deep-think request completely
            final_result = None
            async for progress_update in orchestrator.stream_deep_think(request):
                if progress_update.step == orchestrator.total_steps:
                    final_result = progress_update.details.get('result')
                    break
            
            if final_result:
                # Format the result for storage
                confidence = final_result.get('confidence_score', 0.0)
                confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                
                # Check if we have structured Answer object
                answer_obj = final_result.get('answer')
                
                if answer_obj:
                    # Use structured format matching test file Answer structure
                    formatted_parts = [
                        f"**Deep Think Research Result** {confidence_emoji}",
                        "",
                        "## 📋 Answer",
                        answer_obj.get('content', 'No answer content available'),
                        "",
                        f"**Confidence:** {answer_obj.get('confidence', 0.0):.1%}",
                        ""
                    ]
                    
                    # Add statistics if available
                    if answer_obj.get('statistics'):
                        stats = answer_obj['statistics']
                        formatted_parts.extend([
                            "## 📊 Research Statistics",
                            f"- **Sources analyzed:** {stats.get('sources_count', 0)}",
                            f"- **Key topics:** {', '.join(stats.get('key_topics', []))}",
                            f"- **Research depth:** {stats.get('depth_level', 'Standard')}",
                            ""
                        ])
                    
                    # Add processing time if available
                    if final_result.get('processing_time'):
                        formatted_parts.append(f"*Processing time: {final_result['processing_time']:.1f}s*")
                    
                    formatted_response = "\n".join(formatted_parts)
                    
                    # Store the complete AI response in MongoDB
                    response_doc = {
                        'message_id': str(uuid.uuid4()),
                        'chat_id': chat_id,
                        'user_id': user_id,
                        'message': formatted_response,
                        'timestamp': datetime.utcnow(),
                        'type': 'assistant',
                        'search_results': final_result.get('scraped_content', []),
                        'deepthink_data': final_result,  # Store full deep-think data
                        'shared': False,
                        'deepthink_completed': True  # Mark as completed background task
                    }
                    
                    try:
                        await self.application.mongodb.create_message(response_doc)
                        logger.info(f"✅ Background deep-think response stored successfully for chat_id: {chat_id}")
                    except Exception as e:
                        logger.error(f"❌ Error storing background deep-think response: {e}")
                else:
                    logger.warning(f"⚠️ Background deep-think completed but no structured answer found for chat_id: {chat_id}")
            else:
                logger.error(f"❌ Background deep-think failed to produce result for chat_id: {chat_id}")
                
        except Exception as e:
            logger.error(f"❌ Background deep-think process failed for chat_id: {chat_id}: {e}")
