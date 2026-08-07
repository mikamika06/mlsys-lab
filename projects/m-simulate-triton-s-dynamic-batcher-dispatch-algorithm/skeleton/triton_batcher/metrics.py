def measure_metrics(arrivals: list[int], dispatches: list[dict], compute_us_fn) -> dict:
    """
    Returns {"throughput_req_sec": float, "p99_queue_delay_us": float}
    """
    raise NotImplementedError
