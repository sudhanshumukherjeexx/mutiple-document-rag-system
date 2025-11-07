"""
Metrics collection and monitoring module for RAG system.
Tracks performance, token usage, and quality metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    
    query_id: str
    timestamp: str
    question: str
    
    # Timing metrics
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    guardrail_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    evaluation_latency_ms: float = 0.0
    
    # Retrieval metrics
    documents_retrieved: int = 0
    documents_after_filter: int = 0
    filter_rejection_rate: float = 0.0
    
    # Quality metrics
    final_score: int = 0
    correction_attempts: int = 0
    success: bool = True
    
    # Token usage
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Additional info
    model_used: str = ""
    error_message: Optional[str] = None


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple queries."""
    
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    
    avg_score: float = 0.0
    score_distribution: Dict[int, int] = field(default_factory=dict)
    
    total_tokens: int = 0
    avg_tokens_per_query: float = 0.0
    
    avg_correction_attempts: float = 0.0
    avg_filter_rejection_rate: float = 0.0


class MetricsCollector:
    """Collects and manages metrics for the RAG system."""
    
    def __init__(self, metrics_file: str = "logs/metrics.json", enabled: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            metrics_file: Path to metrics file
            enabled: Whether metrics collection is enabled
        """
        self.metrics_file = metrics_file
        self.enabled = enabled
        self.query_metrics: List[QueryMetrics] = []
        self._lock = Lock()
        
        if enabled:
            Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Metrics collector initialized: {metrics_file}")
    
    def create_query_metrics(self, query_id: str, question: str) -> QueryMetrics:
        """
        Create a new QueryMetrics object.
        
        Args:
            query_id: Unique identifier for the query
            question: The user's question
            
        Returns:
            QueryMetrics instance
        """
        return QueryMetrics(
            query_id=query_id,
            timestamp=datetime.now().isoformat(),
            question=question
        )
    
    def record_query(self, metrics: QueryMetrics) -> None:
        """
        Record metrics for a completed query.
        
        Args:
            metrics: QueryMetrics instance to record
        """
        if not self.enabled:
            return
        
        with self._lock:
            self.query_metrics.append(metrics)
            self._save_to_file()
            logger.debug(f"Recorded metrics for query {metrics.query_id}")
    
    def _save_to_file(self) -> None:
        """Save metrics to JSON file."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "queries": [asdict(m) for m in self.query_metrics]
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")
    
    def get_aggregate_metrics(self) -> AggregateMetrics:
        """
        Calculate aggregate metrics across all queries.
        
        Returns:
            AggregateMetrics instance
        """
        if not self.query_metrics:
            return AggregateMetrics()
        
        metrics = AggregateMetrics()
        metrics.total_queries = len(self.query_metrics)
        
        successful = [m for m in self.query_metrics if m.success]
        failed = [m for m in self.query_metrics if not m.success]
        
        metrics.successful_queries = len(successful)
        metrics.failed_queries = len(failed)
        
        if successful:
            latencies = [m.total_latency_ms for m in successful]
            metrics.avg_latency_ms = sum(latencies) / len(latencies)
            metrics.min_latency_ms = min(latencies)
            metrics.max_latency_ms = max(latencies)
            
            scores = [m.final_score for m in successful]
            metrics.avg_score = sum(scores) / len(scores)
            
            # Score distribution
            for score in scores:
                metrics.score_distribution[score] = metrics.score_distribution.get(score, 0) + 1
            
            metrics.total_tokens = sum(m.total_tokens for m in successful)
            metrics.avg_tokens_per_query = metrics.total_tokens / len(successful)
            
            attempts = [m.correction_attempts for m in successful]
            metrics.avg_correction_attempts = sum(attempts) / len(attempts)
            
            rejection_rates = [m.filter_rejection_rate for m in successful if m.documents_retrieved > 0]
            if rejection_rates:
                metrics.avg_filter_rejection_rate = sum(rejection_rates) / len(rejection_rates)
        
        return metrics
    
    def print_summary(self) -> None:
        """Print a summary of collected metrics."""
        if not self.enabled or not self.query_metrics:
            logger.info("No metrics to display")
            return
        
        agg = self.get_aggregate_metrics()
        
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        print(f"Total Queries: {agg.total_queries}")
        print(f"Successful: {agg.successful_queries}")
        print(f"Failed: {agg.failed_queries}")
        print(f"Success Rate: {(agg.successful_queries/agg.total_queries*100):.1f}%")
        print("\nPerformance:")
        print(f"  Avg Latency: {agg.avg_latency_ms:.0f}ms")
        print(f"  Min Latency: {agg.min_latency_ms:.0f}ms")
        print(f"  Max Latency: {agg.max_latency_ms:.0f}ms")
        print("\nQuality:")
        print(f"  Avg Score: {agg.avg_score:.2f}/5")
        print(f"  Score Distribution: {dict(sorted(agg.score_distribution.items()))}")
        print(f"  Avg Correction Attempts: {agg.avg_correction_attempts:.2f}")
        print("\nToken Usage:")
        print(f"  Total Tokens: {agg.total_tokens:,}")
        print(f"  Avg Tokens/Query: {agg.avg_tokens_per_query:.0f}")
        print("\nFiltering:")
        print(f"  Avg Rejection Rate: {agg.avg_filter_rejection_rate*100:.1f}%")
        print("="*60 + "\n")
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self.query_metrics = []
            logger.info("Metrics cleared")


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = ""):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
    
    def __enter__(self) -> 'Timer':
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        """Stop the timer and calculate elapsed time."""
        end_time = time.perf_counter()
        self.elapsed_ms = (end_time - self.start_time) * 1000
        
        if self.name:
            logger.debug(f"{self.name} took {self.elapsed_ms:.2f}ms")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_metrics(metrics_file: str, enabled: bool) -> MetricsCollector:
    """
    Initialize the global metrics collector.
    
    Args:
        metrics_file: Path to metrics file
        enabled: Whether metrics collection is enabled
        
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    _metrics_collector = MetricsCollector(metrics_file, enabled)
    return _metrics_collector
