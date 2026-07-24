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


def _case(seed, n, d, dv, sigma):
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=sigma, size=d)
    k = rng.normal(scale=sigma, size=(n, d))
    v = rng.normal(size=(n, dv))
    # split into 3-4 chunks of uneven size, like split-KV attention shards
    cuts = sorted(rng.choice(range(1, n), size=min(3, n - 1), replace=False))
    bounds = [0] + list(cuts) + [n]
    partials = [
        _make_partial(q, k[bounds[i]:bounds[i + 1]], v[bounds[i]:bounds[i + 1]])
        for i in range(len(bounds) - 1)
    ]
    ref = _attention_oracle(q, k, v)
    return partials, ref


def grade(sol, fx) -> dict:
    # Large-magnitude logits: realistic enough (bigger-scale keys/queries),
    # and crucially they force each partial's own local max m_i into the
    # hundreds/thousands. The mathematically-equivalent-but-unstable merge
    # (scaling by exp(m_i) instead of exp(m_i - global_max)) overflows to
    # inf/nan at this scale even though it would happen to look correct
    # for small toy logits -- this is the whole point of the rescale.
    cases = [
        _case(1, 20, 32, 4, 20.0),
        _case(2, 15, 24, 6, 25.0),
        _case(3, 30, 40, 3, 18.0),
    ]

    worst = 0.0
    for partials, ref in cases:
        try:
            got = np.asarray(sol.merge_split_kv(list(partials)), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
