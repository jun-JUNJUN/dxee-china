#!/usr/bin/env python3
"""
Research Module - Modular Research System
A comprehensive, pluggable research system with MongoDB caching, enhanced web search,
and AI reasoning capabilities.
"""

from .interfaces import (
    ResearchQuery, SearchResult, ExtractedContent, AnalysisResult, ResearchResult,
    IWebSearchService, IContentExtractor, ICacheService, IAIReasoningService,
    IResearchOrchestrator, IProgressCallback, IMetricsCollector, IConfigurationManager
)

from .config import ConfigurationManager, get_config_manager, reload_config
from .metrics import MetricsCollector, get_metrics_collector, reset_global_metrics
from .cache import MongoDBCacheService, create_cache_service
from .web_search import EnhancedGoogleWebSearchService, create_web_search_service
from .content_extractor import EnhancedWebContentExtractor, create_content_extractor
from .ai_reasoning import DeepSeekReasoningService, create_ai_reasoning_service
from .orchestrator import EnhancedResearchOrchestrator, ProgressTracker, create_research_orchestrator

__version__ = "1.0.0"
__author__ = "Research System Team"

# Export main classes and functions
__all__ = [
    # Data classes
    'ResearchQuery',
    'SearchResult', 
    'ExtractedContent',
    'AnalysisResult',
    'ResearchResult',
    
    # Interfaces
    'IWebSearchService',
    'IContentExtractor', 
    'ICacheService',
    'IAIReasoningService',
    'IResearchOrchestrator',
    'IProgressCallback',
    'IMetricsCollector',
    'IConfigurationManager',
    
    # Implementations
    'ConfigurationManager',
    'MetricsCollector',
    'MongoDBCacheService',
    'EnhancedGoogleWebSearchService',
    'EnhancedWebContentExtractor',
    'DeepSeekReasoningService',
    'EnhancedResearchOrchestrator',
    'ProgressTracker',
    
    # Factory functions
    'create_cache_service',
    'create_web_search_service',
    'create_content_extractor',
    'create_ai_reasoning_service',
    'create_research_orchestrator',
    
    # Utility functions
    'get_config_manager',
    'get_metrics_collector',
    'reload_config',
    'reset_global_metrics',
    
    # Main factory
    'create_research_system'
]


def create_research_system(progress_callback=None, config_overrides=None):
    """
    Factory function to create a complete research system with all components
    
    Args:
        progress_callback: Optional progress callback implementation
        config_overrides: Optional configuration overrides
        
    Returns:
        Tuple of (orchestrator, config, metrics) for the research system
    """
    # Get or create configuration
    config = get_config_manager()
    if config_overrides:
        # Apply configuration overrides if provided
        for key, value in config_overrides.items():
            config._config[key] = value
    
    # Get metrics collector
    metrics = get_metrics_collector()
    
    # Create orchestrator with progress callback
    orchestrator = create_research_orchestrator(progress_callback)
    
    return orchestrator, config, metrics


def get_system_info():
    """Get information about the research system"""
    config = get_config_manager()
    metrics = get_metrics_collector()
    
    return {
        'version': __version__,
        'author': __author__,
        'configuration_summary': config.get_configuration_summary(),
        'metrics_summary': metrics.get_metrics_summary(),
        'services_available': {
            'cache': config.is_service_configured('mongodb'),
            'web_search': config.is_service_configured('google'),
            'ai_reasoning': config.is_service_configured('deepseek')
        }
    }


def validate_system_setup():
    """Validate that the research system is properly configured"""
    config = get_config_manager()
    issues = []
    warnings = []
    
    # Check critical services
    if not config.is_service_configured('deepseek'):
        issues.append("DeepSeek API not configured - AI reasoning will not work")
    
    if not config.is_service_configured('google'):
        warnings.append("Google Search API not configured - web search will be limited")
    
    if not config.is_service_configured('mongodb'):
        warnings.append("MongoDB not configured - caching will be disabled")
    
    # Check model configuration
    model_settings = config.get_model_settings()
    if not model_settings['fallback_models']:
        warnings.append("No fallback models configured")
    
    # Check research settings
    research_settings = config.get_research_settings()
    if research_settings['default_target_relevance'] < 1 or research_settings['default_target_relevance'] > 10:
        warnings.append("Invalid default target relevance score")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'configuration_summary': config.get_configuration_summary()
    }


# Module-level initialization
def _initialize_module():
    """Initialize the research module"""
    try:
        # Validate configuration on import
        validation = validate_system_setup()
        if not validation['valid']:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Research system configuration issues: {validation['issues']}")
            for warning in validation['warnings']:
                logger.warning(f"Research system warning: {warning}")
        
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to initialize research module: {e}")
        return False


# Initialize on import
_module_initialized = _initialize_module()
