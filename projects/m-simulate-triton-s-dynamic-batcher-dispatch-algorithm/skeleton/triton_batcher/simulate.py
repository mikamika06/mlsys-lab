def simulate(arrivals: list[int], max_batch_size: int, preferred: list[int], max_delay_us: int, compute_us_fn) -> list[dict]:
    """
    Simulates Triton's dynamic batching algorithm.
    Returns a list of dicts: {"start_us": int, "batch_size": int, "request_ids": list[int]}
    """
    raise NotImplementedError
