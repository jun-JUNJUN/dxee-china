#!/usr/bin/env python3
"""
Configuration Manager for Research System
Handles all configuration, credentials, and settings
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from .interfaces import IConfigurationManager

logger = logging.getLogger(__name__)


@dataclass
class APICredentials:
    """API credentials for external services"""
    deepseek_api_key: str = ""
    deepseek_api_url: str = "https://api.deepseek.com"
    google_api_key: str = ""
    google_cse_id: str = ""
    mongodb_uri: str = "mongodb://localhost:27017"


@dataclass
class ResearchSettings:
    """Research-specific settings"""
    default_search_mode: str = "standard"
    default_target_relevance: int = 7
    default_max_iterations: int = 3
    default_results_per_query: int = 5
    max_content_length: int = 5000
    extraction_timeout: int = 10
    analysis_timeout: int = 60
    cache_expiry_days: int = 7
    request_delay_seconds: float = 1.0
    max_retries: int = 6


@dataclass
class ModelSettings:
    """AI model settings"""
    primary_chat_model: str = "deepseek-chat"
    primary_reasoning_model: str = "deepseek-reasoner"
    fallback_models: list = field(default_factory=lambda: ["deepseek-chat", "deepseek-reasoner"])
    model_selection_strategy: str = "mode_based"  # "mode_based", "fallback", "round_robin"


@dataclass
class CacheSettings:
    """Cache settings"""
    enabled: bool = True
    database_name: str = "web_research_cache"
    collection_name: str = "scraped_content"
    index_fields: list = field(default_factory=lambda: ["url", "keywords", "accessed_date"])
    max_cache_size_mb: int = 1000
    cleanup_interval_hours: int = 24


class ConfigurationManager(IConfigurationManager):
    """Configuration manager implementation"""
    
    def __init__(self):
        self._config = {}
        self._api_credentials = None
        self._research_settings = None
        self._model_settings = None
        self._cache_settings = None
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from environment variables and defaults"""
        logger.info("Loading configuration from environment variables")
        
        # Load API credentials
        self._api_credentials = APICredentials(
            deepseek_api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
            deepseek_api_url=os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com'),
            google_api_key=os.environ.get('GOOGLE_API_KEY', ''),
            google_cse_id=os.environ.get('GOOGLE_CSE_ID', ''),
            mongodb_uri=os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
        )
        
        # Load research settings
        self._research_settings = ResearchSettings(
            default_search_mode=os.environ.get('DEFAULT_SEARCH_MODE', 'standard'),
            default_target_relevance=int(os.environ.get('DEFAULT_TARGET_RELEVANCE', '7')),
            default_max_iterations=int(os.environ.get('DEFAULT_MAX_ITERATIONS', '3')),
            default_results_per_query=int(os.environ.get('DEFAULT_RESULTS_PER_QUERY', '5')),
            max_content_length=int(os.environ.get('MAX_CONTENT_LENGTH', '5000')),
            extraction_timeout=int(os.environ.get('EXTRACTION_TIMEOUT', '10')),
            analysis_timeout=int(os.environ.get('ANALYSIS_TIMEOUT', '60')),
            cache_expiry_days=int(os.environ.get('CACHE_EXPIRY_DAYS', '7')),
            request_delay_seconds=float(os.environ.get('REQUEST_DELAY_SECONDS', '1.0')),
            max_retries=int(os.environ.get('MAX_RETRIES', '6'))
        )
        
        # Load model settings
        self._model_settings = ModelSettings(
            primary_chat_model=os.environ.get('PRIMARY_CHAT_MODEL', 'deepseek-chat'),
            primary_reasoning_model=os.environ.get('PRIMARY_REASONING_MODEL', 'deepseek-reasoner'),
            fallback_models=os.environ.get('FALLBACK_MODELS', 'deepseek-chat,deepseek-reasoner').split(','),
            model_selection_strategy=os.environ.get('MODEL_SELECTION_STRATEGY', 'mode_based')
        )
        
        # Load cache settings
        self._cache_settings = CacheSettings(
            enabled=os.environ.get('CACHE_ENABLED', 'true').lower() == 'true',
            database_name=os.environ.get('CACHE_DATABASE_NAME', 'web_research_cache'),
            collection_name=os.environ.get('CACHE_COLLECTION_NAME', 'scraped_content'),
            index_fields=os.environ.get('CACHE_INDEX_FIELDS', 'url,keywords,accessed_date').split(','),
            max_cache_size_mb=int(os.environ.get('MAX_CACHE_SIZE_MB', '1000')),
            cleanup_interval_hours=int(os.environ.get('CACHE_CLEANUP_INTERVAL_HOURS', '24'))
        )
        
        # Validate critical configuration
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate critical configuration settings"""
        warnings = []
        errors = []
        
        # Check API credentials
        if not self._api_credentials.deepseek_api_key:
            errors.append("DEEPSEEK_API_KEY is required but not set")
        
        if not self._api_credentials.google_api_key:
            warnings.append("GOOGLE_API_KEY not set - web search functionality will be limited")
        
        if not self._api_credentials.google_cse_id:
            warnings.append("GOOGLE_CSE_ID not set - web search functionality will be limited")
        
        # Check model settings
        if self._model_settings.primary_chat_model not in self._model_settings.fallback_models:
            warnings.append(f"Primary chat model '{self._model_settings.primary_chat_model}' not in fallback models")
        
        if self._model_settings.primary_reasoning_model not in self._model_settings.fallback_models:
            warnings.append(f"Primary reasoning model '{self._model_settings.primary_reasoning_model}' not in fallback models")
        
        # Log warnings and errors
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        for error in errors:
            logger.error(f"Configuration error: {error}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        logger.info("Configuration validation completed successfully")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self._config.get(key, default)
    
    def get_api_credentials(self, service: str = None) -> Dict[str, str]:
        """Get API credentials for a service or all credentials"""
        if service is None:
            return {
                'deepseek_api_key': self._api_credentials.deepseek_api_key,
                'deepseek_api_url': self._api_credentials.deepseek_api_url,
                'google_api_key': self._api_credentials.google_api_key,
                'google_cse_id': self._api_credentials.google_cse_id,
                'mongodb_uri': self._api_credentials.mongodb_uri
            }
        
        service = service.lower()
        if service == 'deepseek':
            return {
                'api_key': self._api_credentials.deepseek_api_key,
                'api_url': self._api_credentials.deepseek_api_url
            }
        elif service == 'google':
            return {
                'api_key': self._api_credentials.google_api_key,
                'cse_id': self._api_credentials.google_cse_id
            }
        elif service == 'mongodb':
            return {
                'uri': self._api_credentials.mongodb_uri
            }
        else:
            raise ValueError(f"Unknown service: {service}")
    
    def get_research_settings(self) -> Dict[str, Any]:
        """Get research-specific settings"""
        return {
            'default_search_mode': self._research_settings.default_search_mode,
            'default_target_relevance': self._research_settings.default_target_relevance,
            'default_max_iterations': self._research_settings.default_max_iterations,
            'default_results_per_query': self._research_settings.default_results_per_query,
            'max_content_length': self._research_settings.max_content_length,
            'extraction_timeout': self._research_settings.extraction_timeout,
            'analysis_timeout': self._research_settings.analysis_timeout,
            'cache_expiry_days': self._research_settings.cache_expiry_days,
            'request_delay_seconds': self._research_settings.request_delay_seconds,
            'max_retries': self._research_settings.max_retries
        }
    
    def get_model_settings(self) -> Dict[str, Any]:
        """Get AI model settings"""
        return {
            'primary_chat_model': self._model_settings.primary_chat_model,
            'primary_reasoning_model': self._model_settings.primary_reasoning_model,
            'fallback_models': self._model_settings.fallback_models,
            'model_selection_strategy': self._model_settings.model_selection_strategy
        }
    
    def get_cache_settings(self) -> Dict[str, Any]:
        """Get cache settings"""
        return {
            'enabled': self._cache_settings.enabled,
            'database_name': self._cache_settings.database_name,
            'collection_name': self._cache_settings.collection_name,
            'index_fields': self._cache_settings.index_fields,
            'max_cache_size_mb': self._cache_settings.max_cache_size_mb,
            'cleanup_interval_hours': self._cache_settings.cleanup_interval_hours
        }
    
    def is_service_configured(self, service: str) -> bool:
        """Check if a service is properly configured"""
        service = service.lower()
        if service == 'deepseek':
            return bool(self._api_credentials.deepseek_api_key)
        elif service == 'google':
            return bool(self._api_credentials.google_api_key and self._api_credentials.google_cse_id)
        elif service == 'mongodb':
            return bool(self._api_credentials.mongodb_uri)
        else:
            return False
    
    def get_model_for_mode(self, search_mode: str) -> str:
        """Get the appropriate model for a search mode"""
        strategy = self._model_settings.model_selection_strategy
        
        if strategy == "mode_based":
            if search_mode in ["deep", "enhanced", "reasoning"]:
                return self._model_settings.primary_reasoning_model
            else:
                return self._model_settings.primary_chat_model
        elif strategy == "fallback":
            return self._model_settings.fallback_models[0]
        elif strategy == "round_robin":
            # Simple round-robin implementation
            import time
            index = int(time.time()) % len(self._model_settings.fallback_models)
            return self._model_settings.fallback_models[index]
        else:
            return self._model_settings.primary_chat_model
    
    def get_fallback_models(self, exclude_model: str = None) -> list:
        """Get fallback models, optionally excluding a specific model"""
        models = self._model_settings.fallback_models.copy()
        if exclude_model and exclude_model in models:
            models.remove(exclude_model)
        return models
    
    def reload_configuration(self):
        """Reload configuration from environment variables"""
        logger.info("Reloading configuration")
        self._load_configuration()
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration (without sensitive data)"""
        return {
            'services_configured': {
                'deepseek': self.is_service_configured('deepseek'),
                'google': self.is_service_configured('google'),
                'mongodb': self.is_service_configured('mongodb')
            },
            'research_settings': {
                'default_search_mode': self._research_settings.default_search_mode,
                'default_target_relevance': self._research_settings.default_target_relevance,
                'default_max_iterations': self._research_settings.default_max_iterations,
                'cache_enabled': self._cache_settings.enabled
            },
            'model_settings': {
                'primary_chat_model': self._model_settings.primary_chat_model,
                'primary_reasoning_model': self._model_settings.primary_reasoning_model,
                'model_selection_strategy': self._model_settings.model_selection_strategy,
                'fallback_models_count': len(self._model_settings.fallback_models)
            }
        }


# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def reload_config():
    """Reload the global configuration"""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload_configuration()
