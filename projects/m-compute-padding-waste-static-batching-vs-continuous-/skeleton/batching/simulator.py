def simulate_batcher(requests, max_batch_size, mode, step_time_ms=10.0):
    """
    Simulates execution of requests under 'static' or 'continuous' batching.

    requests: list of dicts with 'id', 'arrival_time', 'prompt_len', 'decode_len'
    max_batch_size: int
    mode: 'static' or 'continuous'
    step_time_ms: time taken per step per request batch

    Returns dict with:
      - total_time_ms: float
      - throughput_tokens_per_sec: float
      - avg_latency_ms: float
      - avg_ttft_ms: float
      - avg_itl_ms: float
      - request_stats: dict mapping request id to individual timing metrics
    """
    raise NotImplementedError
