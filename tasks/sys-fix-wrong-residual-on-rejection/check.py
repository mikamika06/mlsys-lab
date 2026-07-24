import numpy as np

_PAIRS = [
    (np.array([0.10, 0.05, 0.50, 0.05, 0.20, 0.10]),
     np.array([0.30, 0.30, 0.10, 0.10, 0.10, 0.10])),
    (np.array([0.05, 0.35, 0.05, 0.05, 0.40, 0.10]),
     np.array([0.20, 0.10, 0.30, 0.20, 0.10, 0.10])),
]
_N = 200_000


def _run_speculative(p, q, residual_fn, n, rng):
    """Full accept/reject/resample loop: draft x~q, accept with prob
    min(1, p(x)/q(x)), else resample from residual_fn(p, q)."""
    V = len(p)
    draft = rng.choice(V, size=n, p=q)
    u = rng.random(n)
    accept_prob = np.minimum(1.0, p[draft] / np.maximum(q[draft], 1e-300))
    accept = u < accept_prob
    out = np.where(accept, draft, -1)

    n_reject = int(np.sum(~accept))
    if n_reject > 0:
        r = np.asarray(residual_fn(p, q), dtype=np.float64)
        if r.shape != (V,) or not np.all(np.isfinite(r)) or np.any(r < -1e-9):
            return None
        s = r.sum()
        if s <= 0:
            return None
        r = np.clip(r, 0.0, None)
        r = r / r.sum()
        resampled = rng.choice(V, size=n_reject, p=r)
        out[~accept] = resampled
    return out


def _kl(p, freq, eps=1e-9):
    return float(np.sum(p * np.log((p + eps) / (freq + eps))))


def grade(sol, fx) -> dict:
    """
    Runs the full speculative-decoding accept/reject/resample loop for
    200,000 trials on two fixed (p, q) pairs, using the submission's
    residual_distribution(p, q) for the resample-on-rejection step.
    Compares the empirical output-token frequency to the true target p via
    KL divergence. The correct residual (normalized max(p-q,0)) makes the
    whole scheme unbiased (empirical KL -> 0 as N grows); resampling from p
    directly (the bug) biases the output and produces a much larger KL.
    """
    kls = []
    for p, q in _PAIRS:
        rng = np.random.default_rng(0)
        try:
            out = _run_speculative(p, q, sol.residual_distribution, _N, rng)
        except Exception:
            return {"mean_kl": 999.0}
        if out is None or out.shape != (_N,):
            return {"mean_kl": 999.0}
        V = len(p)
        counts = np.bincount(out, minlength=V).astype(np.float64)
        freq = counts / _N
        kls.append(_kl(p, freq))
    return {"mean_kl": float(np.mean(kls))}
