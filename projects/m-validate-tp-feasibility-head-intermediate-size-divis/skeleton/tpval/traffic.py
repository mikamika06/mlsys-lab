def compute_tp_traffic(config: dict, tp_size: int, target_tokens_per_sec: float) -> dict:
    raise NotImplementedError


def compute_pp_bubble_fraction(num_microbatches: int, num_pipeline_stages: int) -> float:
    raise NotImplementedError
