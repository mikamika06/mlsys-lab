import numpy as np


def _oracle(alpha: float, gamma: int, cost_ratio: float) -> float:
    # E[tokens per round] = sum_{k=0}^{gamma} alpha**k  (== (1-alpha**(gamma+1))/(1-alpha),
    # but the geometric sum avoids the 0/0 issue at alpha == 1 exactly).
    expected_tokens = sum(alpha ** k for k in range(gamma + 1))
    cost_per_round = gamma * cost_ratio + 1.0
    return expected_tokens / cost_per_round


def _gen_case(rng):
    alpha = float(rng.uniform(0.0, 1.0))
    gamma = int(rng.integers(1, 9))
    cost_ratio = float(rng.uniform(0.01, 1.5))
    return alpha, gamma, cost_ratio


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [_gen_case(rng) for _ in range(10)]
    cases += [(0.0, 4, 0.3), (1.0, 4, 0.3), (0.9, 1, 1.0)]

    worst_ratio = 1.0
    for alpha, gamma, cost_ratio in cases:
        expected = _oracle(alpha, gamma, cost_ratio)
        try:
            got = float(sol.speculative_speedup(alpha, gamma, cost_ratio))
        except Exception:
            worst_ratio = 0.0
            break
        if got <= 0 or expected <= 0:
            worst_ratio = 0.0
            break
        ratio = min(got / expected, expected / got)
        worst_ratio = min(worst_ratio, ratio)
    return {"size_ratio": worst_ratio}
