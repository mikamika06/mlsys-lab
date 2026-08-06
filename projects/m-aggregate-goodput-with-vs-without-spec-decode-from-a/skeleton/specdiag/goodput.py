def compute_request_goodput(req_log: dict, penalty_factor: float = 0.5) -> dict:
    """Computes effective goodput for a single request log entry."""
    raise NotImplementedError


def aggregate_goodput_comparison(scheduler_log: list[dict], penalty_factor: float = 0.5) -> dict:
    """Compares aggregate goodput between standard and speculative decoding execution logs."""
    raise NotImplementedError
