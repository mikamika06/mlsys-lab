import numpy as np


def windowed_ring_attention(Q, K, V, W):
    """Sliding-window attention backed by a fixed-size ring buffer.

    Args:
        Q: (n, d)  query rows, streamed one per step.
        K: (n, d)  key rows.
        V: (n, dv) value rows.
        W: int, window size / ring-buffer capacity (1 <= W <= n).

    Stream tokens 0..n-1. Write token t into physical slot (t % W) of a
    capacity-W ring buffer, overwriting (evicting) the oldest token once full.
    At each step attend over the tokens currently in the buffer (the most
    recent min(t+1, W) tokens) with sqrt(d)-scaled, numerically stable softmax.

    Returns (out, Kbuf, Vbuf):
        out  : (n, dv)  per-step windowed-attention outputs
        Kbuf : (W, d)   final physical contents of the key ring buffer
        Vbuf : (W, dv)  final physical contents of the value ring buffer
    """
    raise NotImplementedError("your code here")
