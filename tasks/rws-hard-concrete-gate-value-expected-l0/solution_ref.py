import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hard_concrete_gate(log_alpha: np.ndarray, beta: float, gamma: float, zeta: float):
    """
    Hard-concrete gate (Louizos, Welling & Kingma, "Learning Sparse Neural
    Networks through L0 Regularization"):

    - Deterministic (test-time) gate value:
        z_hat = clip(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)
    - Closed-form expected L0 (probability the gate is nonzero):
        P(z > 0) = sigmoid(log_alpha - beta * log(-gamma/zeta))

    Returns (gate_value, expected_l0), each the same shape as log_alpha.
    """
    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    gate = np.clip(_sigmoid(log_alpha) * (zeta - gamma) + gamma, 0.0, 1.0)
    l0 = _sigmoid(log_alpha - beta * np.log(-gamma / zeta))
    return gate, l0
