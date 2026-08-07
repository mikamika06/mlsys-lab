import numpy as np

def _ref(Q, K, scale=None):
    head_dim = Q.shape[-1]
    used_scale = 1 / np.sqrt(head_dim) if scale is None else float(scale)
    logits = np.matmul(Q, K.transpose(0, 1, 3, 2)) * used_scale
    top_logit = np.max(logits)
    return used_scale, float(top_logit)

def _rel_err(a, b):
    a = float(a)
    b = float(b)
    if abs(b) > 1e-12:
        return abs(a - b) / abs(b)
    else:
        return abs(a - b)

def grade(sol, fx) -> dict:
    np.random.seed(42)
    cases = [
        (np.random.randn(2, 3, 4, 8), np.random.randn(2, 3, 5, 8), None),
        (np.random.randn(1, 1, 6, 16), np.random.randn(1, 1, 7, 16), 0.25),
        (np.random.randn(3, 2, 5, 32), np.random.randn(3, 2, 4, 32), None)
    ]
    max_scale_err = 0.0
    max_logit_err = 0.0
    for Q_np, K_np, scale in cases:
        Q_list = Q_np.tolist()
        K_list = K_np.tolist()
        try:
            got_scale, got_top = sol.measure_sdpa_scale_and_top_logit(Q_list, K_list, scale=scale)
        except Exception:
            return {"scale_rel_err": float("inf"), "logit_rel_err": float("inf")}
        ref_scale, ref_top = _ref(Q_np, K_np, scale)
        scale_err = _rel_err(got_scale, ref_scale)
        logit_err = _rel_err(got_top, ref_top)
        if scale_err > max_scale_err:
            max_scale_err = scale_err
        if logit_err > max_logit_err:
            max_logit_err = logit_err
    return {"scale_rel_err": max_scale_err, "logit_rel_err": max_logit_err}
