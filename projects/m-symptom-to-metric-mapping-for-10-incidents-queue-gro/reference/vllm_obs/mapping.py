"""Symptom-to-metric mapping for vLLM cluster incidents."""

INCIDENT_MAPPINGS = {
    1: {
        "symptom": "Queue growth due to insufficient decode throughput",
        "primary_metric": "vllm:num_requests_waiting",
        "secondary_metric": "vllm:avg_prompt_throughput_tok_s",
        "category": "queue_growth",
    },
    2: {
        "symptom": "Cache thrashing due to fragmentation",
        "primary_metric": "vllm:gpu_cache_usage_perc",
        "secondary_metric": "vllm:num_recomputations_total",
        "category": "cache_thrash",
    },
    3: {
        "symptom": "High preemption rate due to KV memory exhaustion",
        "primary_metric": "vllm:num_preemptions_total",
        "secondary_metric": "vllm:gpu_cache_usage_perc",
        "category": "preemption",
    },
    4: {
        "symptom": "Chunked prefill stall under long context batch",
        "primary_metric": "vllm:time_to_first_token_seconds",
        "secondary_metric": "vllm:num_requests_running",
        "category": "latency",
    },
    5: {
        "symptom": "CPU swapping overflow during heavy preemption",
        "primary_metric": "vllm:cpu_cache_usage_perc",
        "secondary_metric": "vllm:num_preemptions_total",
        "category": "memory",
    },
    6: {
        "symptom": "Worker thread imbalance causing TTFT tail spikes",
        "primary_metric": "vllm:time_to_first_token_seconds",
        "secondary_metric": "vllm:time_in_queue_seconds",
        "category": "latency",
    },
    7: {
        "symptom": "Max model len violation causing immediate request drops",
        "primary_metric": "vllm:request_failure_total",
        "secondary_metric": "vllm:num_requests_waiting",
        "category": "failures",
    },
    8: {
        "symptom": "Continuous request re-execution caused by memory pressure",
        "primary_metric": "vllm:num_recomputations_total",
        "secondary_metric": "vllm:num_preemptions_total",
        "category": "cache_thrash",
    },
    9: {
        "symptom": "Scheduler starvation of long prompts",
        "primary_metric": "vllm:time_in_queue_seconds",
        "secondary_metric": "vllm:num_requests_waiting",
        "category": "queue_growth",
    },
    10: {
        "symptom": "Speculative decoding verification mismatch slowdown",
        "primary_metric": "vllm:inter_token_latency_seconds",
        "secondary_metric": "vllm:avg_generation_throughput_tok_s",
        "category": "latency",
    },
}


def map_incident_to_metric(incident_id: int) -> dict:
    if incident_id not in INCIDENT_MAPPINGS:
        raise ValueError(f"Unknown incident id: {incident_id}")
    return INCIDENT_MAPPINGS[incident_id]


def parse_telemetry_sample(metrics: dict) -> dict:
    waiting = metrics.get("vllm:num_requests_waiting", 0)
    running = metrics.get("vllm:num_requests_running", 0)
    total_reqs = waiting + running
    queue_sat = waiting / total_reqs if total_reqs > 0 else 0.0

    gpu_cache = metrics.get("vllm:gpu_cache_usage_perc", 0.0)
    preemptions = metrics.get("vllm:num_preemptions_total", 0)

    is_thrashing = gpu_cache > 0.90 and preemptions > 5
    is_queue_saturated = queue_sat > 0.50

    return {
        "queue_saturation": round(queue_sat, 4),
        "gpu_cache_usage": gpu_cache,
        "is_thrashing": is_thrashing,
        "is_queue_saturated": is_queue_saturated,
    }
