import numpy as np


def hard_concrete_gate(log_alpha: np.ndarray, beta: float, gamma: float, zeta: float):
    """
    Compute the hard-concrete gate's deterministic (test-time) gate value
    and its closed-form expected L0 (probability of being nonzero), as
    described in task.md.

    Returns (gate_value, expected_l0), each the same shape as log_alpha.
    """
    raise NotImplementedError('your code here')
