import numpy as np
from mlsys import scorers
from mlsys.sim import cache as cachesim


def _softmax_reference(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _trace_reference(shape):
    n, d = shape
    trace = []
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)
    return trace


def _cache_result(addrs):
    result = cachesim.simulate(
        addrs,
        line_bytes=64,
        sets=64,
        ways=2,
    )
    if isinstance(result, dict):
        return result
    return {"misses": int(result)}


def grade(sol, fx) -> dict:
    cases = [
        [
            [1000.0, 1001.0, 1002.0],
            [-1000.0, -999.0, -998.0],
        ],
        [
            [5000.0, 4999.0, 4998.0, 4997.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        [
            [700.0, 701.0, 702.0, 703.0, 704.0],
        ],
    ]

    worst_kl = 0.0
    cache_match = 1.0

    for x in cases:
        x_np = np.asarray(x, dtype=np.float64)
        ref = _softmax_reference(x_np)
        ref_trace = _trace_reference(x_np.shape)

        try:
            got, trace = sol.stable_softmax_kernel(x)
        except Exception:
            return {"mean_kl": 1.0, "cache_match": 0.0}

        worst_kl = max(worst_kl, scorers.mean_kl(ref, np.asarray(got)))

        got_cache = _cache_result(list(trace))
        ref_cache = _cache_result(ref_trace)

        if got_cache != ref_cache:
            cache_match = 0.0

    return {
        "mean_kl": float(worst_kl),
        "cache_match": float(cache_match),
    }
