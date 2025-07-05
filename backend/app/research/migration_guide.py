#!/usr/bin/env python3
"""
Migration Guide and Examples for the Modular Research System
This file provides examples and utilities for migrating from the old research system
to the new modular architecture.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import both old and new systems for comparison
from ..service.deepseek_service import DeepSeekService
from ..service.enhanced_deepseek_service import EnhancedDeepSeekService
from . import (
    create_research_system, ResearchQuery, validate_system_setup,
    get_config_manager, get_metrics_collector
)

logger = logging.getLogger(__name__)


class ResearchSystemMigrator:
    """Utility class to help migrate from old to new research system"""
    
    def __init__(self):
        self.old_service = None
        self.new_service = None
        self.comparison_results = []
    
    async def setup_services(self, input_queue=None, output_queue=None):
        """Setup both old and new services for comparison"""
        try:
            # Setup old service
            if input_queue is None:
                input_queue = []
            if output_queue is None:
                output_queue = []
            
            self.old_service = DeepSeekService(input_queue, output_queue)
            self.new_service = EnhancedDeepSeekService(input_queue, output_queue)
            
            logger.info("✅ Both old and new services initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup services: {e}")
            raise
    
    async def compare_research_methods(self, question: str, search_mode: str = "googleweb") -> Dict[str, Any]:
        """Compare old and new research methods side by side"""
        if not self.old_service or not self.new_service:
            await self.setup_services()
        
        logger.info(f"🔄 Comparing research methods for: {question}")
        
        # Prepare message data
        message_data = {
            'message': question,
            'chat_id': 'comparison_test',
            'message_id': f'test_{int(datetime.utcnow().timestamp())}',
            'search_mode': search_mode
        }
        
        comparison_result = {
            'question': question,
            'search_mode': search_mode,
            'timestamp': datetime.utcnow().isoformat(),
            'old_system': {},
            'new_system': {},
            'comparison': {}
        }
        
        # Test old system
        try:
            logger.info("🔄 Testing old research system...")
            old_start = asyncio.get_event_loop().time()
            old_result = await self.old_service.process_message(message_data)
            old_duration = asyncio.get_event_loop().time() - old_start
            
            comparison_result['old_system'] = {
                'success': True,
                'duration': old_duration,
                'result': old_result,
                'message_length': len(old_result.get('message', '')),
                'search_results_count': len(old_result.get('search_results', [])),
                'has_deep_search_data': 'deep_search_data' in old_result
            }
            
            logger.info(f"✅ Old system completed in {old_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Old system failed: {e}")
            comparison_result['old_system'] = {
                'success': False,
                'error': str(e),
                'duration': 0
            }
        
        # Test new system
        try:
            logger.info("🔄 Testing new research system...")
            new_start = asyncio.get_event_loop().time()
            new_result = await self.new_service.process_message(message_data)
            new_duration = asyncio.get_event_loop().time() - new_start
            
            comparison_result['new_system'] = {
                'success': True,
                'duration': new_duration,
                'result': new_result,
                'message_length': len(new_result.get('message', '')),
                'search_results_count': len(new_result.get('search_results', [])),
                'has_enhanced_data': 'enhanced_research_data' in new_result,
                'research_type': new_result.get('enhanced_research_data', {}).get('research_type'),
                'iterations_completed': new_result.get('enhanced_research_data', {}).get('iterations_completed', 0),
                'final_relevance_score': new_result.get('enhanced_research_data', {}).get('final_relevance_score', 0)
            }
            
            logger.info(f"✅ New system completed in {new_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ New system failed: {e}")
            comparison_result['new_system'] = {
                'success': False,
                'error': str(e),
                'duration': 0
            }
        
        # Generate comparison analysis
        comparison_result['comparison'] = self._analyze_comparison(
            comparison_result['old_system'], 
            comparison_result['new_system']
        )
        
        self.comparison_results.append(comparison_result)
        return comparison_result
    
    def _analyze_comparison(self, old_result: Dict[str, Any], new_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the differences between old and new system results"""
        analysis = {
            'both_successful': old_result.get('success', False) and new_result.get('success', False),
            'performance_improvement': 0,
            'feature_improvements': [],
            'recommendations': []
        }
        
        if old_result.get('success') and new_result.get('success'):
            # Performance comparison
            old_duration = old_result.get('duration', 0)
            new_duration = new_result.get('duration', 0)
            
            if old_duration > 0:
                performance_change = ((old_duration - new_duration) / old_duration) * 100
                analysis['performance_improvement'] = performance_change
                
                if performance_change > 10:
                    analysis['recommendations'].append(f"New system is {performance_change:.1f}% faster")
                elif performance_change < -10:
                    analysis['recommendations'].append(f"New system is {abs(performance_change):.1f}% slower")
            
            # Feature comparison
            if new_result.get('has_enhanced_data'):
                analysis['feature_improvements'].append("Enhanced research data with metrics")
            
            if new_result.get('iterations_completed', 0) > 0:
                analysis['feature_improvements'].append("Iterative research with relevance scoring")
            
            if new_result.get('research_type'):
                analysis['feature_improvements'].append(f"Research type classification: {new_result.get('research_type')}")
            
            # Content quality comparison
            old_length = old_result.get('message_length', 0)
            new_length = new_result.get('message_length', 0)
            
            if new_length > old_length * 1.2:
                analysis['feature_improvements'].append("More comprehensive analysis")
            
            # Search results comparison
            old_results = old_result.get('search_results_count', 0)
            new_results = new_result.get('search_results_count', 0)
            
            if new_results > old_results:
                analysis['feature_improvements'].append(f"More search results ({new_results} vs {old_results})")
        
        elif new_result.get('success') and not old_result.get('success'):
            analysis['recommendations'].append("New system succeeded where old system failed")
        elif old_result.get('success') and not new_result.get('success'):
            analysis['recommendations'].append("Old system succeeded where new system failed - investigate")
        
        return analysis
    
    def get_migration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive migration report"""
        if not self.comparison_results:
            return {'error': 'No comparison results available'}
        
        total_comparisons = len(self.comparison_results)
        successful_old = sum(1 for r in self.comparison_results if r['old_system'].get('success'))
        successful_new = sum(1 for r in self.comparison_results if r['new_system'].get('success'))
        
        avg_performance_improvement = sum(
            r['comparison'].get('performance_improvement', 0) 
            for r in self.comparison_results
        ) / total_comparisons
        
        all_features = []
        for result in self.comparison_results:
            all_features.extend(result['comparison'].get('feature_improvements', []))
        
        unique_features = list(set(all_features))
        
        return {
            'summary': {
                'total_comparisons': total_comparisons,
                'old_system_success_rate': (successful_old / total_comparisons) * 100,
                'new_system_success_rate': (successful_new / total_comparisons) * 100,
                'average_performance_improvement': avg_performance_improvement
            },
            'feature_improvements': unique_features,
            'detailed_results': self.comparison_results,
            'migration_recommendation': self._get_migration_recommendation(
                successful_old, successful_new, total_comparisons, avg_performance_improvement
            )
        }
    
    def _get_migration_recommendation(self, successful_old: int, successful_new: int, 
                                    total: int, avg_performance: float) -> str:
        """Generate migration recommendation based on comparison results"""
        new_success_rate = (successful_new / total) * 100
        old_success_rate = (successful_old / total) * 100
        
        if new_success_rate >= old_success_rate and avg_performance >= 0:
            return "RECOMMENDED: Migrate to new system - equal or better performance and reliability"
        elif new_success_rate > old_success_rate * 0.9 and avg_performance > 10:
            return "RECOMMENDED: Migrate to new system - significant performance improvement"
        elif new_success_rate < old_success_rate * 0.8:
            return "NOT RECOMMENDED: New system has lower success rate - investigate issues"
        else:
            return "CONDITIONAL: Consider gradual migration with monitoring"


async def example_basic_usage():
    """Example of basic usage of the new research system"""
    logger.info("🚀 Example: Basic usage of new research system")
    
    try:
        # Validate system setup
        validation = validate_system_setup()
        if not validation['valid']:
            logger.error(f"System validation failed: {validation['issues']}")
            return
        
        # Create research system
        orchestrator, config, metrics = create_research_system()
        
        # Create a research query
        query = ResearchQuery(
            question="What are the top CRM software companies in Japan by revenue?",
            query_id="example_001",
            timestamp=datetime.utcnow(),
            search_mode="enhanced",
            target_relevance=7,
            max_iterations=2
        )
        
        logger.info(f"🔍 Starting research: {query.question}")
        
        # Conduct research
        result = await orchestrator.conduct_research(query)
        
        # Display results
        if result.success:
            logger.info("✅ Research completed successfully!")
            logger.info(f"📊 Research type: {result.research_type}")
            
            if result.metrics:
                logger.info(f"🎯 Target achieved: {result.metrics.get('target_achieved', False)}")
                logger.info(f"📈 Final relevance: {result.metrics.get('final_relevance_score', 0)}/10")
                logger.info(f"🔄 Iterations: {result.metrics.get('iterations_completed', 0)}")
            
            if result.analysis:
                logger.info(f"📝 Analysis length: {len(result.analysis.analysis_content)} characters")
                logger.info(f"🧠 Has reasoning: {bool(result.analysis.reasoning_content)}")
            
            if result.extracted_contents:
                successful = sum(1 for c in result.extracted_contents if c.success)
                logger.info(f"📄 Content extracted: {successful}/{len(result.extracted_contents)} successful")
        else:
            logger.error(f"❌ Research failed: {result.error}")
        
        # Cleanup
        await orchestrator.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")


async def example_comparison_test():
    """Example of comparing old vs new research systems"""
    logger.info("🔄 Example: Comparing old vs new research systems")
    
    try:
        migrator = ResearchSystemMigrator()
        
        # Test questions
        test_questions = [
            "What is the capital of France?",  # Simple question
            "Find the top 5 CRM software companies in Japan by revenue",  # Complex research
            "Explain machine learning algorithms"  # Knowledge-based question
        ]
        
        for question in test_questions:
            logger.info(f"\n🧪 Testing: {question}")
            
            # Compare both systems
            comparison = await migrator.compare_research_methods(question, "googleweb")
            
            # Display comparison results
            old_success = comparison['old_system'].get('success', False)
            new_success = comparison['new_system'].get('success', False)
            
            logger.info(f"  Old system: {'✅' if old_success else '❌'} ({comparison['old_system'].get('duration', 0):.2f}s)")
            logger.info(f"  New system: {'✅' if new_success else '❌'} ({comparison['new_system'].get('duration', 0):.2f}s)")
            
            if comparison['comparison']['feature_improvements']:
                logger.info(f"  Improvements: {', '.join(comparison['comparison']['feature_improvements'])}")
        
        # Generate migration report
        report = migrator.get_migration_report()
        logger.info(f"\n📊 Migration Report:")
        logger.info(f"  Recommendation: {report['migration_recommendation']}")
        logger.info(f"  Performance improvement: {report['summary']['average_performance_improvement']:.1f}%")
        logger.info(f"  New system success rate: {report['summary']['new_system_success_rate']:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Comparison test failed: {e}")


async def example_service_integration():
    """Example of integrating the new service with existing backend"""
    logger.info("🔗 Example: Service integration with existing backend")
    
    try:
        # Create enhanced service (as would be done in the backend)
        input_queue = []
        output_queue = []
        
        enhanced_service = EnhancedDeepSeekService(input_queue, output_queue)
        
        # Test message processing
        test_message = {
            'message': 'What are the latest trends in AI research?',
            'chat_id': 'test_chat_001',
            'message_id': 'msg_001',
            'search_mode': 'enhanced'
        }
        
        logger.info("📨 Processing test message...")
        result = await enhanced_service.process_message(test_message)
        
        logger.info("✅ Message processed successfully!")
        logger.info(f"📝 Response length: {len(result.get('message', ''))} characters")
        logger.info(f"🔍 Search results: {len(result.get('search_results', []))}")
        
        if 'enhanced_research_data' in result:
            enhanced_data = result['enhanced_research_data']
            logger.info(f"🎯 Research type: {enhanced_data.get('research_type')}")
            logger.info(f"📊 Relevance score: {enhanced_data.get('final_relevance_score', 0)}/10")
        
        # Get system status
        status = enhanced_service.get_system_status()
        logger.info(f"🖥️ System status: {status['system_initialized']}")
        
        # Cleanup
        await enhanced_service.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Service integration example failed: {e}")


if __name__ == "__main__":
    # Run examples
    async def main():
        logger.info("🚀 Running migration guide examples")
        
        await example_basic_usage()
        await example_comparison_test()
        await example_service_integration()
        
        logger.info("✅ All examples completed!")
    
    asyncio.run(main())
