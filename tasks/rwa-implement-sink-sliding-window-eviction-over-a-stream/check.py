import numpy as np
from mlsys import scorers


def _oracle(Q, K, V, k, w):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape
    outputs = []
    lengths = []
    for t in range(n):
        sinks = list(range(min(k, t + 1)))
        start = max(0, t - w + 1)
        recent = list(range(start, t + 1))
        kept = sorted(set(sinks + recent))
        lengths.append(len(kept))

        logits = (K[kept] @ Q[t]) / np.sqrt(d)
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / np.sum(weights)
        outputs.append(weights @ V[kept])
    return np.asarray(outputs, dtype=np.float64), lengths


def grade(sol, fx) -> dict:
    cases = [
        (6, 3, 2, 1, 2),
        (8, 4, 3, 2, 3),
        (5, 2, 2, 0, 2),
    ]
    max_err = 0.0
    cache_ok = 1.0
    rng = np.random.default_rng(1234)

    for n, d, m, k, w in cases:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, m))
        ref_out, ref_lengths = _oracle(Q, K, V, k, w)

        try:
            got_out, got_lengths = sol.sink_attention_stream(Q, K, V, k, w)
        except Exception:
            return {"max_abs_err": float("inf"), "cache_bound": 0.0}

        err = scorers.max_abs_err(ref_out, got_out)
        max_err = max(max_err, err)

        if list(got_lengths) != ref_lengths:
            cache_ok = 0.0

        if any(int(x) > k + w for x in got_lengths):
            cache_ok = 0.0

    return {
        "max_abs_err": float(max_err),
        "cache_bound": cache_ok,
    }
