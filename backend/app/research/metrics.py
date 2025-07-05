#!/usr/bin/env python3
"""
Metrics Collection System for Research Components
Tracks performance, usage, and quality metrics
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from .interfaces import IMetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class TimingMetric:
    """Represents a timing metric"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CounterMetric:
    """Represents a counter metric"""
    name: str
    value: int
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GaugeMetric:
    """Represents a gauge metric"""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HistogramMetric:
    """Represents a histogram metric"""
    name: str
    values: List[float] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_value(self, value: float):
        """Add a value to the histogram"""
        self.values.append(value)
    
    def get_percentile(self, percentile: float) -> float:
        """Get a percentile value"""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_average(self) -> float:
        """Get average value"""
        return sum(self.values) / len(self.values) if self.values else 0.0


class MetricsCollector(IMetricsCollector):
    """Metrics collector implementation"""
    
    def __init__(self, max_history_size: int = 10000):
        self._timings: Dict[str, TimingMetric] = {}
        self._counters: Dict[str, CounterMetric] = defaultdict(lambda: CounterMetric("", 0))
        self._gauges: Dict[str, GaugeMetric] = {}
        self._histograms: Dict[str, HistogramMetric] = defaultdict(lambda: HistogramMetric(""))
        self._max_history_size = max_history_size
        self._timing_history: deque = deque(maxlen=max_history_size)
        self._metric_history: deque = deque(maxlen=max_history_size)
        self._session_start = time.time()
        
        logger.info(f"MetricsCollector initialized with max history size: {max_history_size}")
    
    def start_timing(self, operation: str, tags: Dict[str, str] = None) -> str:
        """Start timing an operation, return timing ID"""
        timing_id = f"{operation}_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        timing = TimingMetric(
            operation=operation,
            start_time=start_time,
            tags=tags or {}
        )
        
        self._timings[timing_id] = timing
        logger.debug(f"Started timing: {operation} (ID: {timing_id})")
        
        return timing_id
    
    def end_timing(self, timing_id: str) -> float:
        """End timing an operation, return duration"""
        if timing_id not in self._timings:
            logger.warning(f"Timing ID not found: {timing_id}")
            return 0.0
        
        timing = self._timings[timing_id]
        end_time = time.time()
        duration = end_time - timing.start_time
        
        timing.end_time = end_time
        timing.duration = duration
        
        # Add to history
        self._timing_history.append(timing)
        
        # Update histogram
        histogram_key = f"timing_{timing.operation}"
        self._histograms[histogram_key].name = histogram_key
        self._histograms[histogram_key].add_value(duration)
        self._histograms[histogram_key].tags.update(timing.tags)
        
        logger.debug(f"Ended timing: {timing.operation} (Duration: {duration:.3f}s)")
        
        # Clean up
        del self._timings[timing_id]
        
        return duration
    
    def record_metric(self, name: str, value: Any, tags: Dict[str, str] = None):
        """Record a metric"""
        tags = tags or {}
        timestamp = datetime.utcnow()
        
        if isinstance(value, (int, float)):
            if name.startswith("counter_"):
                # Counter metric
                counter_key = f"{name}_{hash(frozenset(tags.items()))}"
                if counter_key in self._counters:
                    self._counters[counter_key].value += value
                else:
                    self._counters[counter_key] = CounterMetric(name, value, tags, timestamp)
            else:
                # Gauge metric
                gauge_key = f"{name}_{hash(frozenset(tags.items()))}"
                self._gauges[gauge_key] = GaugeMetric(name, float(value), tags, timestamp)
                
                # Also add to histogram for trend analysis
                histogram_key = f"histogram_{name}"
                self._histograms[histogram_key].name = histogram_key
                self._histograms[histogram_key].add_value(float(value))
                self._histograms[histogram_key].tags.update(tags)
        
        # Add to general metric history
        self._metric_history.append({
            'name': name,
            'value': value,
            'tags': tags,
            'timestamp': timestamp
        })
        
        logger.debug(f"Recorded metric: {name} = {value} (tags: {tags})")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.record_metric(f"counter_{name}", value, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        self.record_metric(name, value, tags)
    
    def record_histogram_value(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a value in a histogram"""
        tags = tags or {}
        histogram_key = f"histogram_{name}_{hash(frozenset(tags.items()))}"
        
        if histogram_key not in self._histograms:
            self._histograms[histogram_key] = HistogramMetric(name, [], tags)
        
        self._histograms[histogram_key].add_value(value)
        logger.debug(f"Recorded histogram value: {name} = {value}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        current_time = time.time()
        session_duration = current_time - self._session_start
        
        # Timing summaries
        timing_summaries = {}
        for histogram_name, histogram in self._histograms.items():
            if histogram_name.startswith("timing_") and histogram.values:
                operation = histogram_name.replace("timing_", "")
                timing_summaries[operation] = {
                    'count': len(histogram.values),
                    'total_time': sum(histogram.values),
                    'avg_time': histogram.get_average(),
                    'min_time': min(histogram.values),
                    'max_time': max(histogram.values),
                    'p50': histogram.get_percentile(50),
                    'p95': histogram.get_percentile(95),
                    'p99': histogram.get_percentile(99)
                }
        
        # Counter summaries
        counter_summaries = {}
        for counter in self._counters.values():
            if counter.name:
                counter_summaries[counter.name] = {
                    'value': counter.value,
                    'tags': counter.tags
                }
        
        # Gauge summaries
        gauge_summaries = {}
        for gauge in self._gauges.values():
            gauge_summaries[gauge.name] = {
                'value': gauge.value,
                'tags': gauge.tags,
                'timestamp': gauge.timestamp.isoformat()
            }
        
        # Histogram summaries
        histogram_summaries = {}
        for histogram_name, histogram in self._histograms.items():
            if not histogram_name.startswith("timing_") and histogram.values:
                histogram_summaries[histogram.name] = {
                    'count': len(histogram.values),
                    'avg': histogram.get_average(),
                    'min': min(histogram.values),
                    'max': max(histogram.values),
                    'p50': histogram.get_percentile(50),
                    'p95': histogram.get_percentile(95),
                    'p99': histogram.get_percentile(99)
                }
        
        return {
            'session_duration': session_duration,
            'session_start': datetime.fromtimestamp(self._session_start).isoformat(),
            'metrics_collected': len(self._metric_history),
            'timings_recorded': len(self._timing_history),
            'active_timings': len(self._timings),
            'timing_summaries': timing_summaries,
            'counter_summaries': counter_summaries,
            'gauge_summaries': gauge_summaries,
            'histogram_summaries': histogram_summaries
        }
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_metrics = [
            metric for metric in self._metric_history
            if metric['timestamp'] >= cutoff_time
        ]
        
        recent_timings = [
            timing for timing in self._timing_history
            if timing.end_time and datetime.fromtimestamp(timing.end_time) >= cutoff_time
        ]
        
        return {
            'time_window_minutes': minutes,
            'recent_metrics_count': len(recent_metrics),
            'recent_timings_count': len(recent_timings),
            'recent_metrics': recent_metrics[-50:],  # Last 50 metrics
            'recent_timings': [
                {
                    'operation': timing.operation,
                    'duration': timing.duration,
                    'tags': timing.tags
                }
                for timing in recent_timings[-20:]  # Last 20 timings
            ]
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report"""
        summary = self.get_metrics_summary()
        recent = self.get_recent_metrics(10)  # Last 10 minutes
        
        # Calculate performance insights
        insights = []
        
        # Check for slow operations
        for operation, stats in summary.get('timing_summaries', {}).items():
            if stats['avg_time'] > 5.0:  # Slower than 5 seconds
                insights.append(f"Operation '{operation}' is slow (avg: {stats['avg_time']:.2f}s)")
            
            if stats['p95'] > stats['avg_time'] * 3:  # High variance
                insights.append(f"Operation '{operation}' has high variance (p95: {stats['p95']:.2f}s)")
        
        # Check counter trends
        for counter_name, counter_data in summary.get('counter_summaries', {}).items():
            if counter_data['value'] > 100:
                insights.append(f"High counter value for '{counter_name}': {counter_data['value']}")
        
        return {
            'summary': summary,
            'recent_activity': recent,
            'performance_insights': insights,
            'report_generated_at': datetime.utcnow().isoformat()
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self._timings.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timing_history.clear()
        self._metric_history.clear()
        self._session_start = time.time()
        
        logger.info("All metrics have been reset")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format.lower() == "json":
            import json
            return json.dumps(self.get_metrics_summary(), indent=2, default=str)
        elif format.lower() == "csv":
            # Simple CSV export for timing data
            lines = ["operation,duration,timestamp,tags"]
            for timing in self._timing_history:
                if timing.duration:
                    tags_str = ";".join([f"{k}={v}" for k, v in timing.tags.items()])
                    lines.append(f"{timing.operation},{timing.duration},{timing.end_time},{tags_str}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics collector instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def reset_global_metrics():
    """Reset the global metrics collector"""
    global _metrics_collector
    if _metrics_collector is not None:
        _metrics_collector.reset_metrics()
