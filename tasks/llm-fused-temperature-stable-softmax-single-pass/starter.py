import numpy as np


def fused_softmax(logits, T):
    """Temperature-scaled, numerically-stable softmax over a 1-D row of logits.

    Return ``softmax(logits / T)`` as a float64 NumPy array, computed in a SINGLE
    pass over the row: fuse the temperature division into one online reduction that
    tracks a running max and a running normalizer (rescaling the normalizer by
    ``exp(m_old - m_new)`` whenever the max grows), then normalize. Do NOT make
    separate max / sum / divide passes with Python loops, and do NOT exponentiate
    the raw scaled logits (that overflows).
    """
    raise NotImplementedError("your code here")
