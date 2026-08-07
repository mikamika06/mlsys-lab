import math
import numpy as np


def cascade_stage1_accept(q1: np.ndarray, q2: np.ndarray, x1: int, rng: np.random.Generator) -> tuple[bool, int]:
    prob = min(1.0, float(q2[x1] / q1[x1]))
    if rng.uniform() < prob:
        return True, int(x1)
    res = np.maximum(0.0, q2 - q1)
    s = np.sum(res)
    if s < 1e-12:
        p_res = q2 / np.sum(q2)
    else:
        p_res = res / s
    x2 = int(rng.choice(len(q2), p=p_res))
    return False, x2


def cascade_stage2_accept(q2: np.ndarray, p: np.ndarray, x2: int, rng: np.random.Generator) -> tuple[bool, int]:
    prob = min(1.0, float(p[x2] / q2[x2]))
    if rng.uniform() < prob:
        return True, int(x2)
    res = np.maximum(0.0, p - q2)
    s = np.sum(res)
    if s < 1e-12:
        p_res = p / np.sum(p)
    else:
        p_res = res / s
    x_final = int(rng.choice(len(p), p=p_res))
    return False, x_final


def multi_draft_select(candidates: list[int], q_drafts: list[np.ndarray], p: np.ndarray, rng: np.random.Generator) -> tuple[bool, int, int]:
    k_drafts = len(q_drafts)
    q_mix = np.mean(q_drafts, axis=0)
    idx = int(rng.integers(0, k_drafts))
    cand_tok = candidates[idx]
    prob = min(1.0, float(p[cand_tok] / q_mix[cand_tok]))
    if rng.uniform() < prob:
        return True, idx, int(cand_tok)
    res = np.maximum(0.0, p - q_mix)
    s = np.sum(res)
    if s < 1e-12:
        p_res = p / np.sum(p)
    else:
        p_res = res / s
    x_final = int(rng.choice(len(p), p=p_res))
    return False, -1, x_final


def expected_tokens(alpha: float, gamma: int) -> float:
    if math.isclose(alpha, 1.0):
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def cascade_latency_per_token(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha2: float) -> float:
    cost = c1 * gamma1 + c2 * gamma2 + cT
    tokens = expected_tokens(alpha2, gamma2)
    return float(cost / tokens)


def is_2stage_net_win(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha2: float, alpha_direct: float) -> bool:
    l_cascade = cascade_latency_per_token(c1, gamma1, c2, gamma2, cT, alpha2)
    cost_1stage = c1 * gamma1 + cT
    tokens_1stage = expected_tokens(alpha_direct, gamma1)
    l_1stage = cost_1stage / tokens_1stage
    return bool(l_cascade < l_1stage)


def break_even_alpha2(c1: float, gamma1: int, c2: float, gamma2: int, cT: float, alpha_direct: float) -> float:
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


DISTRIBUTIONS = [
    (np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.1, 0.5, 0.3, 0.1]), np.array([0.2, 0.2, 0.2, 0.4])),
    (np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.2, 0.6, 0.1, 0.1]), np.array([0.1, 0.1, 0.1, 0.7])),
    (np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.1, 0.4, 0.4, 0.1]), np.array([0.05, 0.05, 0.8, 0.1]))
]

LATENCY_CONFIGS = [
    (1.0, 5, 2.0, 3, 20.0, 0.7, 0.5),
    (0.5, 8, 1.5, 4, 15.0, 0.85, 0.6),
    (2.0, 3, 3.0, 2, 30.0, 0.4, 0.3),
    (0.8, 6, 1.0, 5, 10.0, 0.9, 0.75)
]
