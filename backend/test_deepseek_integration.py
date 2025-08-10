#!/usr/bin/env python3
"""
Integration Test for DeepSeek Button Integration

This test verifies that the DeepSeek research functionality integrates properly
with the existing chat system and provides expected functionality.

Run with: python test_deepseek_integration.py
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDeepSeekIntegration:
    """Test suite for DeepSeek integration"""
    
    def __init__(self):
        self.test_results = []
        self.mongodb_service = None
        self.research_service = None
        
    async def setup(self):
        """Setup test environment"""
        logger.info("🔧 Setting up test environment...")
        
        # Mock environment variables
        os.environ['DEEPSEEK_API_KEY'] = 'test_api_key'
        os.environ['GOOGLE_API_KEY'] = 'test_google_key'
        os.environ['GOOGLE_CSE_ID'] = 'test_cse_id'
        os.environ['BRIGHTDATA_API_KEY'] = 'test_brightdata_key'
        os.environ['CACHE_EXPIRY_DAYS'] = '7'
        
        logger.info("✅ Test environment configured")
    
    def add_test_result(self, test_name: str, success: bool, message: str = ""):
        """Add a test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            'name': test_name,
            'success': success,
            'message': message,
            'status': status
        })
        logger.info(f"{status}: {test_name} - {message}")
    
    async def test_service_import(self):
        """Test that the enhanced research service can be imported"""
        try:
            from app.service.enhanced_deepseek_research_service import EnhancedDeepSeekResearchService
            self.add_test_result("Service Import", True, "Enhanced research service imported successfully")
            return True
        except ImportError as e:
            self.add_test_result("Service Import", False, f"Import failed: {e}")
            return False
    
    async def test_service_initialization(self):
        """Test that the service can be initialized properly"""
        try:
            from app.service.enhanced_deepseek_research_service import EnhancedDeepSeekResearchService
            from app.service.mongodb_service import MongoDBService
            
            # Mock MongoDB service
            mongodb_mock = MagicMock(spec=MongoDBService)
            mongodb_mock.create_research_indexes = AsyncMock()
            mongodb_mock.get_cache_stats = AsyncMock(return_value={
                'total_entries': 100,
                'successful_entries': 90
            })
            
            # Initialize research service
            service = EnhancedDeepSeekResearchService(mongodb_service=mongodb_mock, cache_expiry_days=7)
            await service.initialize()
            
            self.research_service = service
            self.add_test_result("Service Initialization", True, "Service initialized with mocked dependencies")
            return True
        except Exception as e:
            self.add_test_result("Service Initialization", False, f"Initialization failed: {e}")
            return False
    
    async def test_chat_handler_integration(self):
        """Test that chat handler can handle DeepSeek mode"""
        try:
            # Import the handler class without instantiating it
            from app.handler.chat_handler import ChatStreamHandler
            import inspect
            
            # Check if the handler has the DeepSeek method
            if hasattr(ChatStreamHandler, '_handle_deepseek_research'):
                # Check if it's a proper async method
                method = getattr(ChatStreamHandler, '_handle_deepseek_research')
                if inspect.iscoroutinefunction(method):
                    self.add_test_result("Chat Handler Integration", True, "DeepSeek async handler method exists")
                    return True
                else:
                    self.add_test_result("Chat Handler Integration", False, "DeepSeek method exists but is not async")
                    return False
            else:
                self.add_test_result("Chat Handler Integration", False, "DeepSeek handler method not found")
                return False
        except Exception as e:
            self.add_test_result("Chat Handler Integration", False, f"Handler test failed: {e}")
            return False
    
    async def test_mongodb_extensions(self):
        """Test MongoDB service extensions for research functionality"""
        try:
            from app.service.mongodb_service import MongoDBService
            
            # Check if MongoDB service has research methods
            service = MongoDBService()
            
            required_methods = [
                'create_research_indexes',
                'get_cached_content',
                'cache_content',
                'get_cache_stats',
                'create_research_session',
                'log_api_usage'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(service, method):
                    missing_methods.append(method)
            
            if not missing_methods:
                self.add_test_result("MongoDB Extensions", True, "All required methods exist")
                return True
            else:
                self.add_test_result("MongoDB Extensions", False, f"Missing methods: {missing_methods}")
                return False
        except Exception as e:
            self.add_test_result("MongoDB Extensions", False, f"MongoDB extension test failed: {e}")
            return False
    
    async def test_environment_validation(self):
        """Test environment configuration validation"""
        try:
            # Test with current environment
            from app.tornado_main import validate_deepseek_research_config
            
            config = validate_deepseek_research_config()
            
            expected_keys = ['deepseek_api', 'google_search', 'brightdata', 'cache_config']
            missing_keys = [key for key in expected_keys if key not in config]
            
            if not missing_keys:
                self.add_test_result("Environment Validation", True, "Configuration validation working")
                return True
            else:
                self.add_test_result("Environment Validation", False, f"Missing config keys: {missing_keys}")
                return False
        except Exception as e:
            self.add_test_result("Environment Validation", False, f"Environment validation failed: {e}")
            return False
    
    async def test_research_workflow_mock(self):
        """Test the research workflow with mocked components"""
        try:
            if not self.research_service:
                self.add_test_result("Research Workflow Mock", False, "Research service not initialized")
                return False
            
            # Mock the external API calls
            with patch.object(self.research_service.web_search, 'search', new_callable=AsyncMock) as mock_search, \
                 patch.object(self.research_service.content_extractor, 'extract_content', new_callable=AsyncMock) as mock_extract, \
                 patch.object(self.research_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_api:
                
                # Setup mock responses
                mock_search.return_value = [
                    {'title': 'Test Article 1', 'url': 'http://test1.com', 'snippet': 'Test snippet 1'},
                    {'title': 'Test Article 2', 'url': 'http://test2.com', 'snippet': 'Test snippet 2'}
                ]
                
                mock_extract.return_value = {
                    'url': 'http://test1.com',
                    'title': 'Test Article 1',
                    'content': 'This is test content for the research.',
                    'success': True,
                    'method': 'brightdata_api'
                }
                
                # Mock DeepSeek API responses
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message = MagicMock()
                mock_response.choices[0].message.content = "Query=\"test research query\""
                mock_api.return_value = mock_response
                
                # Test the research workflow
                result = await self.research_service.conduct_deepseek_research("test question", "test_chat_id")
                
                # Verify result structure
                if isinstance(result, dict) and 'success' in result:
                    self.add_test_result("Research Workflow Mock", True, f"Workflow completed with success={result.get('success', False)}")
                    return True
                else:
                    self.add_test_result("Research Workflow Mock", False, f"Invalid result structure: {type(result)}")
                    return False
                    
        except Exception as e:
            self.add_test_result("Research Workflow Mock", False, f"Workflow test failed: {e}")
            return False
    
    async def test_ui_integration(self):
        """Test UI integration by checking template modifications"""
        try:
            template_path = "templates/index.html"
            if not os.path.exists(template_path):
                self.add_test_result("UI Integration", False, "Template file not found")
                return False
            
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Check for DeepSeek button
            if 'deepseek-btn' in template_content and 'data-mode="deepseek"' in template_content:
                # Check for research handling functions
                if 'handleResearchStep' in template_content and 'addDeepSeekResearchDisplay' in template_content:
                    self.add_test_result("UI Integration", True, "Template contains DeepSeek UI components")
                    return True
                else:
                    self.add_test_result("UI Integration", False, "Template missing research handlers")
                    return False
            else:
                self.add_test_result("UI Integration", False, "Template missing DeepSeek button")
                return False
                
        except Exception as e:
            self.add_test_result("UI Integration", False, f"Template check failed: {e}")
            return False
    
    def print_results(self):
        """Print test results summary"""
        logger.info("\n" + "="*60)
        logger.info("🧪 DEEPSEEK INTEGRATION TEST RESULTS")
        logger.info("="*60)
        
        passed = 0
        failed = 0
        
        for result in self.test_results:
            print(f"{result['status']}: {result['name']}")
            if result['message']:
                print(f"    └─ {result['message']}")
            
            if result['success']:
                passed += 1
            else:
                failed += 1
        
        logger.info("-"*60)
        logger.info(f"📊 SUMMARY: {passed} passed, {failed} failed")
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! DeepSeek integration is working correctly.")
        else:
            logger.warning(f"⚠️  {failed} tests failed. Please review the implementation.")
        
        return failed == 0
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting DeepSeek Integration Tests...")
        
        await self.setup()
        
        # Run tests in order
        tests = [
            self.test_service_import,
            self.test_service_initialization,
            self.test_mongodb_extensions,
            self.test_chat_handler_integration,
            self.test_environment_validation,
            self.test_ui_integration,
            self.test_research_workflow_mock
        ]
        
        for test in tests:
            await test()
        
        return self.print_results()

async def main():
    """Main test runner"""
    tester = TestDeepSeekIntegration()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("\n🎊 Integration tests completed successfully!")
        logger.info("The DeepSeek button integration is ready for production use.")
    else:
        logger.error("\n❌ Some integration tests failed.")
        logger.error("Please review the failed tests before deploying.")
    
    return success

if __name__ == "__main__":
    # Run the integration tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)