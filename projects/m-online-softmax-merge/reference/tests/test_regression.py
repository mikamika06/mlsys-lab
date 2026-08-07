import numpy as np
from onlinesoftmax.merge import merge_online_softmax, chunked_online_attention
from onlinesoftmax.harness import compute_rel_err, verify_tolerance_bounds, analyze_error_vs_seqlen


def test_online_softmax_merge_accuracy():
    rng = np.random.default_rng(123)
    q = rng.normal(size=(8, 32))
    k = rng.normal(size=(256, 32))
    v = rng.normal(size=(256, 32))

    got = chunked_online_attention(q, k, v, chunk_size=32)

    scale = 1.0 / np.sqrt(32)
    scores = (q @ k.T) * scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - scores_max)
    want = (exp_s / np.sum(exp_s, axis=-1, keepdims=True)) @ v

    res = verify_tolerance_bounds(got, want, rtol=1e-10, atol=1e-10)
    assert res["passed"], f"Relative error too high: {res['max_rel_err']}"


def test_error_scaling_trend():
    seqlens = [64, 128, 256, 512]
    res = analyze_error_vs_seqlen(16, seqlens, chunk_size=32, seed=42)
    for n in seqlens:
        assert res[n] < 1e-12, f"Excessive error for seqlen {n}: {res[n]}"
