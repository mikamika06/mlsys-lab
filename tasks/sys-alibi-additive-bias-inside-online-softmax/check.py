import numpy as np

def _numpy_reference(scores, slopes):
    """Vectorised reference: add ALiBi bias, then row-wise softmax."""
    n = scores.shape[0]
    i_idx = np.arange(n)[:, None]
    j_idx = np.arange(n)[None, :]
    bias = slopes[:, None] * (i_idx - j_idx)
    biased = scores.astype(np.float64) + bias
    row_max = biased.max(axis=1, keepdims=True)
    exp_shifted = np.exp(biased - row_max)
    ref = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    return ref

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(0)
    test_cases = [
        (rng.randn(4, 4), np.array([0.0625, 0.0625, 0.0625, 0.0625])),
        (rng.randn(8, 8), np.array([0.0625, 0.03125, 0.015625, 0.0078125,
                                     0.0625, 0.03125, 0.015625, 0.0078125])),
        (rng.randn(3, 3), np.array([1.0, 0.5, 0.25])),
        (np.zeros((5, 5)), np.ones(5) * 1.0),
        (rng.randn(16, 16), np.full(16, 0.0625)),
    ]

    max_err = 0.0
    for scores, slopes in test_cases:
        ref = _numpy_reference(scores, slopes)
        try:
            got = np.asarray(
                sol.alibi_online_softmax(scores.copy(), slopes.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": 1.0}
        if got.shape != ref.shape:
            return {"max_abs_err": 1.0}
        err = float(np.max(np.abs(got - ref)))
        max_err = max(max_err, err)
    return {"max_abs_err": max_err}
