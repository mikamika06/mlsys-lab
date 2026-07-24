import numpy as np

def _ref(W, X):
    weight_scales = np.linalg.norm(W, axis=1)
    activation_scales = np.linalg.norm(X, axis=1)
    return weight_scales, activation_scales

def grade(sol, fx) -> dict:
    cases = [
        (np.random.randn(4, 5), np.random.randn(3, 5)),
        (np.random.randn(10, 20), np.random.randn(7, 20)),
        (np.random.randn(1, 8), np.random.randn(2, 8)),
        (np.random.randn(6, 6), np.random.randn(6, 6)),
    ]
    max_err = 0.0
    for W, X in cases:
        try:
            got_weight_scales, got_activation_scales = sol.torchao_scale_granularities(W, X)
        except Exception:
            return {"rel_err": 1.0}
        ref_w, ref_a = _ref(W, X)
        got = np.concatenate([got_weight_scales.ravel(), got_activation_scales.ravel()])
        ref = np.concatenate([ref_w.ravel(), ref_a.ravel()])
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
