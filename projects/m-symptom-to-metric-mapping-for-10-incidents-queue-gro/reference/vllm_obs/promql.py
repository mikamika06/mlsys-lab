"""PromQL query templates for vLLM metrics."""


def get_p95_ttft_query(window: str = "5m") -> str:
    return f"histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[{window}])) by (le))"


def get_kv_utilization_query() -> str:
    return "vllm:gpu_cache_usage_perc"


def get_waiting_queue_saturation_query() -> str:
    return "vllm:num_requests_waiting / (vllm:num_requests_waiting + vllm:num_requests_running)"


def get_preemptions_per_minute_query(window: str = "5m") -> str:
    return f"rate(vllm:num_preemptions_total[{window}]) * 60"
