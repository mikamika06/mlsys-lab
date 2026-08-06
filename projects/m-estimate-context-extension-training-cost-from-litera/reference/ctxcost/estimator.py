def estimate_cost(base_tokens: int, target_length: int, base_length: int, alpha: float) -> float:
    ratio = float(target_length) / float(base_length)
    return float(base_tokens) * (ratio ** alpha)
