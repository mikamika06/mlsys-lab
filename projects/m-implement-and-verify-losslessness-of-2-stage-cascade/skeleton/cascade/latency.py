def expected_tokens(alpha: float, gamma: int) -> float:
    raise NotImplementedError


def cascade_latency_per_token(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha2: float) -> float:
    raise NotImplementedError


def is_2stage_net_win(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha2: float, alpha_direct: float) -> bool:
    raise NotImplementedError


def break_even_alpha2(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha_direct: float) -> float:
    raise NotImplementedError
