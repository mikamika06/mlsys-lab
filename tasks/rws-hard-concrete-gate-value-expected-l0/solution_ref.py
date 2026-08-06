import math
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


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
    shape = log_alpha.shape
    flat_log = log_alpha.flatten()
    gate_list = []
    l0_list = []
    log_term = math.log(-gamma / zeta)
    for x in flat_log:
        sig = _sigmoid(x)
        g = sig * (zeta - gamma) + gamma
        if g < 0.0:
            g = 0.0
        elif g > 1.0:
            g = 1.0
        gate_list.append(g)
        l0_list.append(_sigmoid(x - beta * log_term))
    gate = np.array(gate_list, dtype=np.float64).reshape(shape)
    l0 = np.array(l0_list, dtype=np.float64).reshape(shape)
    return gate, l0
