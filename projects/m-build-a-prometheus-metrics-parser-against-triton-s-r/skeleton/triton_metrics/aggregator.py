from typing import Dict, List
from triton_metrics.parser import MetricSample


def compute_model_request_summary(samples: List[MetricSample]) -> Dict[str, Dict[str, float]]:
    """
    Summarize request counts and compute latency for each model.
    Returns dict mapping model_name -> {'success_count': float, 'avg_compute_time_ms': float}.
    """
    raise NotImplementedError


def compute_gpu_utilization_summary(samples: List[MetricSample]) -> Dict[str, float]:
    """
    Summarize average GPU utilization per GPU device/UUID.
    Returns dict mapping gpu_id -> avg_utilization_percentage.
    """
    raise NotImplementedError
