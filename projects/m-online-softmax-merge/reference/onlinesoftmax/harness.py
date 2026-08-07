import numpy as np
from onlinesoftmax.merge import chunked_online_attention


def compute_rel_err(got, want):
    denom = np.maximum(np.abs(want), 1e-12)
    return float(np.max(np.abs(got - want) / denom))


def verify_tolerance_bounds(got, want, rtol=1e-5, atol=1e-8):
    diff = np.abs(got - want)
    bound = atol + rtol * np.abs(want)
    passed = bool(np.all(diff <= bound))
    max_rel = compute_rel_err(got, want)
    return {"passed": passed, "max_rel_err": max_rel}


def analyze_error_vs_seqlen(query_dim, seqlens, chunk_size=64, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    d_k = query_dim
    d_v = query_dim

    for n in seqlens:
        q = rng.normal(0.0, 1.0, size=(16, d_k)).astype(np.float64)
        k = rng.normal(0.0, 1.0, size=(n, d_k)).astype(np.float64)
        v = rng.normal(0.0, 1.0, size=(n, d_v)).astype(np.float64)

        scale = 1.0 / np.sqrt(d_k)
        scores = (q @ k.T) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        exact_out = probs @ v

        chunk_out = chunked_online_attention(q, k, v, chunk_size=chunk_size)
        err = compute_rel_err(chunk_out, exact_out)
        results[n] = err

    return results
