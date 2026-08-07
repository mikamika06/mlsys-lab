import math

def biased_flash_backward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], B: list[list[float]], dO: list[list[float]], m: list[float], l: list[float]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Memory-efficient attention backward with an additive bias (e.g.
    ALiBi or a mask bias), given only the saved row statistics (m, l)
    from the forward pass -- never a cached probability matrix.

    Q, K, V   : (n, d)
    B         : (n, n) additive bias, added to the scaled scores before
                softmax on the forward pass. Fixed (no gradient wanted).
    dO        : (n, d) upstream gradient w.r.t. the forward output O.
    m, l      : (n,) row max and row softmax-normalizer saved during the
                forward pass, i.e. S = Q@K.T/sqrt(d) + B,
                m = rowmax(S), l = rowsum(exp(S - m)).

    Returns (dQ, dK, dV), each shaped like Q, K, V respectively.
    """
    raise NotImplementedError('your code here')
