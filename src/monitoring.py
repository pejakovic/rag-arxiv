from prometheus_client import Counter, Histogram, start_http_server
import time
from functools import wraps
from typing import Callable, Any

# Metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries processed')
QUERY_ERRORS = Counter('rag_query_errors_total', 'Total number of query errors')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Time spent processing queries')
DOC_RETRIEVAL_DURATION = Histogram('rag_doc_retrieval_seconds', 'Time spent retrieving documents')
MODEL_INFERENCE_DURATION = Histogram('rag_model_inference_seconds', 'Time spent on model inference')

def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server"""
    start_http_server(port)

def monitor_time(metric: Histogram) -> Callable:
    """Decorator to monitor execution time of a function"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.observe(duration)
        return wrapper
    return decorator

def monitor_errors(func: Callable) -> Callable:
    """Decorator to monitor function errors"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            QUERY_ERRORS.inc()
            raise
    return wrapper 