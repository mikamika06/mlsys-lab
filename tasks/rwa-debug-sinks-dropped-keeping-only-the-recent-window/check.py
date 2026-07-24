import numpy as np
from mlsys import scorers

def _correct_sink_keeping(q, k, v, window_size):
    T, d = q.shape
    out = []
    for t in range(T):
        q_t = q[t:t+1]
        if t == 0:
            indices = [0]
        else:
            start = max(1, t - window_size + 2)
            indices = [0] + list(range(start, t + 1))
        K = k[indices]
        V = v[indices]
        scores = q_t @ K.T / np.sqrt(d)
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=1, keepdims=True)
        out.append(weights @ V)
    return np.vstack(out)

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(20241001)
    T = 8
    d = 4
    window_size = 4

    q = rng.randn(T, d).astype(np.float64)
    k = rng.randn(T, d).astype(np.float64)
    v = rng.randn(T, d).astype(np.float64)

    # Emphasise the sink token: key=0 ensures stable attention, value is large
    k[0] = 0.0
    v[0] = 10.0

    try:
        student_out = np.asarray(sol.streaming_attention(q, k, v, window_size))
    except Exception:
        return {"max_abs_err": 1.0}

    ref_out = _correct_sink_keeping(q, k, v, window_size)
    err = float(scorers.max_abs_err(ref_out, student_out))
    return {"max_abs_err": err}
