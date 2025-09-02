#!/usr/bin/env python3
"""
Comprehensive Error Handling and Recovery System for Deep-Think
Implements circuit breakers, exponential backoff, health monitoring, and graceful degradation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DegradationLevel(Enum):
    """System degradation levels"""
    FULL = "full"
    MINIMAL_DEGRADATION = "minimal_degradation"
    SIGNIFICANT_DEGRADATION = "significant_degradation"
    EMERGENCY_MODE = "emergency_mode"


class CircuitBreaker:
    """Circuit breaker pattern implementation for external services"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if the circuit allows execution"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half-open"
                self.half_open_calls = 0
                return True
            return False
        elif self.state == "half-open":
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful execution"""
        if self.state == "half-open":
            self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker reset to closed state")
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "half-open":
            self.state = "open"
            logger.warning(f"Circuit breaker opened after half-open failure")
        elif self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
        if self.state == "half-open":
            self.half_open_calls += 1


class ExponentialBackoff:
    """Exponential backoff strategy for retries"""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, max_retries: int = 3):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        if attempt >= self.max_retries:
            return None  # No more retries
        
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay
    
    async def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                delay = self.calculate_delay(attempt)
                if delay is None:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for {func.__name__}")
                    raise e
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s")
                await asyncio.sleep(delay)
        
        # This shouldn't be reached, but just in case
        raise Exception(f"All retry attempts failed for {func.__name__}")


class HealthMonitor:
    """System health monitoring and reporting"""
    
    def __init__(self):
        self.services = {}
        self.health_history = {}
    
    def register_service(self, service_name: str):
        """Register a service for monitoring"""
        self.services[service_name] = {
            'status': ServiceHealth.UNKNOWN,
            'last_check': None,
            'consecutive_failures': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'circuit_breaker': CircuitBreaker()
        }
    
    def update_service_health(self, service_name: str, success: bool, response_time: float = None):
        """Update service health based on request result"""
        if service_name not in self.services:
            self.register_service(service_name)
        
        service = self.services[service_name]
        service['last_check'] = datetime.now()
        service['total_requests'] += 1
        
        if success:
            service['successful_requests'] += 1
            service['consecutive_failures'] = 0
            service['circuit_breaker'].record_success()
            
            # Update status based on success rate
            success_rate = service['successful_requests'] / service['total_requests']
            if success_rate >= 0.95:
                service['status'] = ServiceHealth.HEALTHY
            elif success_rate >= 0.8:
                service['status'] = ServiceHealth.DEGRADED
            else:
                service['status'] = ServiceHealth.UNHEALTHY
        else:
            service['consecutive_failures'] += 1
            service['circuit_breaker'].record_failure()
            
            if service['consecutive_failures'] >= 3:
                service['status'] = ServiceHealth.UNHEALTHY
            elif service['consecutive_failures'] >= 2:
                service['status'] = ServiceHealth.DEGRADED
    
    def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get current health status of a service"""
        if service_name not in self.services:
            return ServiceHealth.UNKNOWN
        return self.services[service_name]['status']
    
    def can_use_service(self, service_name: str) -> bool:
        """Check if service can be used (not circuit broken)"""
        if service_name not in self.services:
            return True  # Unknown services are assumed available
        return self.services[service_name]['circuit_breaker'].can_execute()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.services:
            return {'level': DegradationLevel.EMERGENCY_MODE, 'services': {}}
        
        healthy_count = sum(1 for s in self.services.values() if s['status'] == ServiceHealth.HEALTHY)
        total_count = len(self.services)
        health_ratio = healthy_count / total_count
        
        if health_ratio >= 1.0:
            level = DegradationLevel.FULL
        elif health_ratio >= 0.75:
            level = DegradationLevel.MINIMAL_DEGRADATION
        elif health_ratio >= 0.5:
            level = DegradationLevel.SIGNIFICANT_DEGRADATION
        else:
            level = DegradationLevel.EMERGENCY_MODE
        
        return {
            'level': level,
            'health_ratio': health_ratio,
            'services': {name: service['status'].value for name, service in self.services.items()}
        }


class ErrorRecoverySystem:
    """Main error recovery and degradation system"""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.backoff_strategy = ExponentialBackoff()
        self.fallback_handlers = {}
        
        # Register core services
        self.health_monitor.register_service('deepseek_api')
        self.health_monitor.register_service('serper_api')
        self.health_monitor.register_service('mongodb')
        
        logger.info("Error recovery system initialized")
    
    def register_fallback(self, service_name: str, fallback_func: Callable):
        """Register fallback function for a service"""
        self.fallback_handlers[service_name] = fallback_func
        logger.info(f"Fallback handler registered for {service_name}")
    
    async def execute_with_recovery(self, service_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with comprehensive error recovery"""
        start_time = time.time()
        
        # Check if service is available via circuit breaker
        if not self.health_monitor.can_use_service(service_name):
            logger.warning(f"Service {service_name} is circuit broken, using fallback")
            return await self._execute_fallback(service_name, *args, **kwargs)
        
        try:
            # Try with exponential backoff
            result = await self.backoff_strategy.retry_with_backoff(func, *args, **kwargs)
            
            # Record success
            response_time = time.time() - start_time
            self.health_monitor.update_service_health(service_name, True, response_time)
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            self.health_monitor.update_service_health(service_name, False, response_time)
            
            logger.error(f"Service {service_name} failed after retries: {e}")
            
            # Try fallback
            return await self._execute_fallback(service_name, *args, **kwargs)
    
    async def _execute_fallback(self, service_name: str, *args, **kwargs) -> Any:
        """Execute fallback handler for a service"""
        if service_name in self.fallback_handlers:
            try:
                logger.info(f"Executing fallback for {service_name}")
                return await self.fallback_handlers[service_name](*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback for {service_name} also failed: {e}")
        
        # Ultimate fallback - return empty/default result
        logger.warning(f"No fallback available for {service_name}, returning default")
        return self._get_default_response(service_name)
    
    def _get_default_response(self, service_name: str) -> Any:
        """Get default response for a service when all else fails"""
        defaults = {
            'deepseek_api': {'content': '{"analysis": "fallback", "confidence": 0.7}'},
            'serper_api': [],
            'mongodb': None
        }
        return defaults.get(service_name, None)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        overall_health = self.health_monitor.get_overall_health()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'degradation_level': overall_health['level'].value,
            'health_ratio': overall_health['health_ratio'],
            'services': overall_health['services'],
            'recommendations': self._get_recommendations(overall_health['level'])
        }
    
    def _get_recommendations(self, degradation_level: DegradationLevel) -> List[str]:
        """Get recommendations based on degradation level"""
        recommendations = {
            DegradationLevel.FULL: ["System operating normally"],
            DegradationLevel.MINIMAL_DEGRADATION: [
                "Some services degraded - monitor closely",
                "Consider reducing request rate"
            ],
            DegradationLevel.SIGNIFICANT_DEGRADATION: [
                "Multiple services impacted - enable fallback modes",
                "Alert operations team",
                "Consider maintenance window"
            ],
            DegradationLevel.EMERGENCY_MODE: [
                "Critical system degradation - immediate attention required",
                "Enable all fallback mechanisms",
                "Escalate to emergency response team"
            ]
        }
        return recommendations.get(degradation_level, ["Unknown status - investigate immediately"])


# Global error recovery system instance
error_recovery = ErrorRecoverySystem()