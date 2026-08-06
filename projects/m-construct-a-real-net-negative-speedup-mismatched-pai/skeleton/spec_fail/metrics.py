import numpy as np

def expected_acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the expected speculative decoding acceptance rate (alpha)
    given target probabilities p and draft probabilities q.
    """
    raise NotImplementedError

def expected_speedup(p: np.ndarray, q: np.ndarray, gamma: int, t_draft: float, t_target: float) -> float:
    """
    Calculate the theoretical speedup of speculative decoding over autoregressive decoding.
    """
    raise NotImplementedError

def acceptance_collapse(p_in: np.ndarray, q_in: np.ndarray, p_out: np.ndarray, q_out: np.ndarray) -> float:
    """
    Calculate the difference in acceptance rate (out_domain - in_domain).
    """
    raise NotImplementedError
