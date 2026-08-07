import math

def flash_attention_backward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], O: list[list[float]], L: list[float], dO: list[list[float]], scale: float):
    """
    Recompute-based FlashAttention backward: recompute P from Q, K, L
    (never read a stored full attention matrix), then apply the softmax
    VJP to get dQ, dK, dV.

    BUG: this drops the D_i = rowsum(dO * O) correction term when forming
    dS, so dQ and dK come out wrong for any row where P isn't one-hot.
    Fix it.
    """
    raise NotImplementedError('your code here')
