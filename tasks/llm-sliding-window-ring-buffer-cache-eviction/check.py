import numpy as np

from mlsys import scorers


def _windowed_reference(Q, K, V, W):
    """Straightforward sliding-window causal attention (real oracle).

    At step t the query attends to the most recent W keys/values, indices
    max(0, t - W + 1) .. t, in natural chronological order. No ring buffer,
    no clever indexing — just the definition.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape
    scale = np.sqrt(d)
    out = np.empty((n, V.shape[1]), dtype=np.float64)
    for t in range(n):
        lo = max(0, t - W + 1)
        idx = np.arange(lo, t + 1)
        logits = (K[idx] @ Q[t]) / scale
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / np.sum(p)
        out[t] = p @ V[idx]
    return out


def _ring_layout(K, V, W):
    """Final physical contents of a capacity-W ring buffer that writes token t
    into slot (t % W). Last writer wins. Test cases use n > W so every slot is
    overwritten by one of the last W tokens (buffer fully populated).
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = K.shape
    Kbuf = np.zeros((W, d), dtype=np.float64)
    Vbuf = np.zeros((W, V.shape[1]), dtype=np.float64)
    for t in range(n):
        Kbuf[t % W] = K[t]
        Vbuf[t % W] = V[t]
    return Kbuf, Vbuf


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    # (n, d, dv, W); every case keeps n > W so the ring buffer fully wraps.
    cases = [
        (10, 4, 3, 4),
        (7, 3, 2, 3),
        (12, 5, 4, 5),
        (16, 4, 4, 6),
        (9, 2, 3, 1),
    ]
    out_err = 0.0
    buf_err = 0.0

    for n, d, dv, W in cases:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, dv))

        ref_out = _windowed_reference(Q, K, V, W)
        ref_Kbuf, ref_Vbuf = _ring_layout(K, V, W)

        Q_list = Q.tolist()
        K_list = K.tolist()
        V_list = V.tolist()

        try:
            got_out, got_Kbuf, got_Vbuf = sol.windowed_ring_attention(Q_list, K_list, V_list, W)
            got_out = np.asarray(got_out, dtype=np.float64)
            got_Kbuf = np.asarray(got_Kbuf, dtype=np.float64)
            got_Vbuf = np.asarray(got_Vbuf, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "buffer_max_abs_err": float("inf")}

        if (got_out.shape != ref_out.shape
                or got_Kbuf.shape != ref_Kbuf.shape
                or got_Vbuf.shape != ref_Vbuf.shape):
            return {"max_abs_err": float("inf"), "buffer_max_abs_err": float("inf")}

        out_err = max(out_err, scorers.max_abs_err(ref_out, got_out))
        buf_err = max(
            buf_err,
            scorers.max_abs_err(ref_Kbuf, got_Kbuf),
            scorers.max_abs_err(ref_Vbuf, got_Vbuf),
        )

    return {
        "max_abs_err": float(out_err),
        "buffer_max_abs_err": float(buf_err),
    }
