import math


def expected_tokens(alpha: float, gamma: int) -> float:
    """Compute expected generated tokens for spec depth gamma and accept rate alpha."""
    if math.isclose(alpha, 1.0):
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def cascade_latency_per_token(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha2: float) -> float:
    """Compute expected per-token latency for a 2-stage cascade."""
    cost = c1 * gamma1 + c2 * gamma2 + cT
    tokens = expected_tokens(alpha2, gamma2)
    return float(cost / tokens)


def is_2stage_net_win(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha2: float, alpha_direct: float) -> bool:
    """Determine if 2-stage cascade beats 1-stage speculation in per-token latency."""
    l_cascade = cascade_latency_per_token(c1, gamma1, c2, gamma2, cT, alpha2)
    cost_1stage = c1 * gamma1 + cT
    tokens_1stage = expected_tokens(alpha_direct, gamma1)
    l_1stage = cost_1stage / tokens_1stage
    return bool(l_cascade < l_1stage)


def break_even_alpha2(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha_direct: float) -> float:
    """Find minimum target acceptance rate alpha2 needed for 2-stage cascade to be a net win."""
    cost_1stage = c1 * gamma1 + cT
    tokens_1stage = expected_tokens(alpha_direct, gamma1)
    l_1stage = cost_1stage / tokens_1stage

    cost_cascade = c1 * gamma1 + c2 * gamma2 + cT
    req_tokens = cost_cascade / l_1stage

    if req_tokens > (gamma2 + 1):
        return 1.0
    if req_tokens <= 1.0:
        return 0.0

    low, high = 0.0, 1.0
    for _ in range(50):
        mid = (low + high) / 2.0
        if expected_tokens(mid, gamma2) >= req_tokens:
            high = mid
        else:
            low = mid
    return float(high)
