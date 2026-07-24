import numpy as np


def _attention_oracle(q, k, v):
    logits = k @ q
    m = np.max(logits)
    weights = np.exp(logits - m)
    weights = weights / np.sum(weights)
    return weights @ v


def _make_partial(q, k, v):
    logits = k @ q
    m = np.max(logits)
    scaled = np.exp(logits - m)
    l = np.sum(scaled)
    o = scaled @ v
    return float(m), float(l), np.asarray(o, dtype=np.float64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    q = rng.normal(size=16).astype(np.float64)
    k = rng.normal(size=(12, 16)).astype(np.float64)
    v = rng.normal(size=(12, 8)).astype(np.float64)

    partials = [
        _make_partial(q, k[:3], v[:3]),
        _make_partial(q, k[3:7], v[3:7]),
        _make_partial(q, k[7:10], v[7:10]),
        _make_partial(q, k[10:], v[10:]),
    ]

    try:
        got = np.asarray(sol.merge_split_kv(partials), dtype=np.float64)
    except Exception:
        return {"max_abs_err": float("inf")}

    ref = _attention_oracle(q, k, v)
    return {"max_abs_err": float(np.max(np.abs(got - ref)))}
