import math

import numpy as np

N_HEADS_LIST = [1, 2, 3, 5, 6, 8, 11, 16, 20, 24, 32]


def _power_of_2_slopes(n):
    start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
    ratio = start
    return [start * (ratio ** i) for i in range(n)]


def _slopes_list(n_heads):
    if math.log2(n_heads).is_integer():
        return _power_of_2_slopes(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    extra = _slopes_list(2 * closest)[0::2][: n_heads - closest]
    return _power_of_2_slopes(closest) + extra


def alibi_slopes(n_heads):
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    return np.array(_slopes_list(n_heads), dtype=np.float64)


def softcap_forward(x, cap):
    x = np.asarray(x, dtype=np.float64)
    return cap * np.tanh(x / cap)


def softcap_backward(grad_output, x, cap):
    x = np.asarray(x, dtype=np.float64)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    t = np.tanh(x / cap)
    return grad_output * (1.0 - t * t)


def _make_softcap_cases():
    rng = np.random.default_rng(0)
    cases = []
    for cap in (1.0, 5.0, 20.0, 0.5):
        edge = np.array(
            [0.0, cap, -cap, 2.0 * cap, -2.0 * cap, 3.0 * cap, -3.0 * cap],
            dtype=np.float64,
        )
        random_part = rng.uniform(-4.0 * cap, 4.0 * cap, size=64)
        x = np.concatenate([edge, random_part])
        cases.append((x, cap))
    return cases


SOFTCAP_CASES = _make_softcap_cases()
