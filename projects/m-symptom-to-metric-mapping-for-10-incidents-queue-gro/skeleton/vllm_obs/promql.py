"""PromQL query templates for vLLM metrics."""


def get_p95_ttft_query(window: str = "5m") -> str:
    raise NotImplementedError


def get_kv_utilization_query() -> str:
    raise NotImplementedError


def get_waiting_queue_saturation_query() -> str:
    raise NotImplementedError


def get_preemptions_per_minute_query(window: str = "5m") -> str:
    raise NotImplementedError
