import numpy as np


def windowed_ring_attention(Q, K, V, W):
    """Sliding-window attention backed by a fixed-size ring buffer.

    Streams tokens 0..n-1. Token t is written to physical slot (t % W) of a
    capacity-W ring buffer, evicting (overwriting) the oldest token once full.
    At each step the query attends over the tokens currently in the buffer
    (the most recent min(t+1, W) tokens). Softmax is order-invariant, so the
    scrambled physical order in the buffer does not affect the output.

    Returns (out, Kbuf, Vbuf):
      out  : (n, dv)  per-step windowed-attention outputs
      Kbuf : (W, d)   final physical contents of the key ring buffer
      Vbuf : (W, dv)  final physical contents of the value ring buffer
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    dv = V.shape[1]
    scale = np.sqrt(d)

    Kbuf = np.zeros((W, d), dtype=np.float64)
    Vbuf = np.zeros((W, dv), dtype=np.float64)
    out = np.empty((n, dv), dtype=np.float64)

    filled = 0  # number of live slots = min(t + 1, W)
    for t in range(n):
        slot = t % W
        Kbuf[slot] = K[t]
        Vbuf[slot] = V[t]
        filled = min(filled + 1, W)

        # Before the first wrap the live tokens occupy slots 0..filled-1;
        # after wrapping all W slots are live. Either way Kbuf[:filled] is
        # exactly the current window (in physical order).
        keys = Kbuf[:filled]
        vals = Vbuf[:filled]

        logits = (keys @ Q[t]) / scale
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / np.sum(p)
        out[t] = p @ vals

    return out, Kbuf, Vbuf
