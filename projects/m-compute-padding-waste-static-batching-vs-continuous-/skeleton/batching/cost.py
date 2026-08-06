def compute_cost_efficiency(logs, gpu_hourly_cost):
    """
    Computes throughput per GPU dollar from recorded serving logs.

    logs: list of dicts with 'mode', 'total_useful_tokens', 'execution_time_sec'
    gpu_hourly_cost: float

    Returns dict mapping mode ('static', 'continuous') to:
      - total_tokens: int
      - total_cost_dollars: float
      - tokens_per_dollar: float
    """
    raise NotImplementedError
