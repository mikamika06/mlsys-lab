import numpy as np

def expected_acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.minimum(p, q)))

def expected_speedup(p: np.ndarray, q: np.ndarray, gamma: int, t_draft: float, t_target: float) -> float:
    alpha = expected_acceptance_rate(p, q)
    if alpha >= 0.999999:
        expected_tokens = float(gamma + 1)
    else:
        expected_tokens = (1.0 - alpha**(gamma + 1)) / (1.0 - alpha)

    step_time = gamma * t_draft + t_target
    return (expected_tokens / step_time) * t_target

def acceptance_collapse(p_in: np.ndarray, q_in: np.ndarray, p_out: np.ndarray, q_out: np.ndarray) -> float:
    rate_in = expected_acceptance_rate(p_in, q_in)
    rate_out = expected_acceptance_rate(p_out, q_out)
    return rate_out - rate_in
