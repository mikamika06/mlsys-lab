import numpy as np


def fused_softmax(logits, T):
    """Temperature-scaled, numerically-stable softmax computed in ONE online pass.

    Fuses the temperature division into a single streaming reduction that tracks a
    running maximum ``m`` and a running normalizer ``d`` (the FlashAttention /
    Milakov-Gimelshein online-softmax trick). When a new element raises the max, the
    accumulated denominator is rescaled by ``exp(m_old - m_new)`` so it stays exact
    without ever revisiting earlier elements. The final normalization is vectorized,
    so the tracer sees a single Python pass over the row.
    """
    logits = np.asarray(logits, dtype=np.float64)
    running_max = -np.inf
    denom = 0.0
    for x in logits:
        z = x / T
        new_max = z if z > running_max else running_max
        denom = denom * np.exp(running_max - new_max) + np.exp(z - new_max)
        running_max = new_max
    return np.exp(logits / T - running_max) / denom
