def evaluate_slo(requests: list[dict], slo_ttft_ms: float, slo_tpot_ms: float, duration_s: float) -> dict:
    """Evaluate SLO compliance and compute throughput vs goodput."""
    raise NotImplementedError


def rank_configs(configs: list[dict], slo_ttft_ms: float, slo_tpot_ms: float) -> list[dict]:
    """Rank serving configurations by goodput."""
    raise NotImplementedError
