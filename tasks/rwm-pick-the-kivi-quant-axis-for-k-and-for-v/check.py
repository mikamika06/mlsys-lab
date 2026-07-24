import numpy as np

def _axis_label(arr: np.ndarray) -> str:
    # population variance (ddof=0)
    channel_var = np.var(arr, axis=0, ddof=0).sum()
    token_var   = np.var(arr, axis=1, ddof=0).sum()
    return "channel" if channel_var <= token_var else "token"

def grade(sol, fx) -> dict:
    # deterministic test cases
    K1 = np.array([[0, 0], [10, 10]])          # token better
    V1 = np.array([[0, 5], [0, 5]])            # channel better

    K2 = np.random.default_rng(42).normal(size=(4, 3))
    V2 = np.random.default_rng(43).normal(size=(5, 6))

    cases = [(K1, V1), (K2, V2)]

    ok = 1.0
    for K, V in cases:
        try:
            got_k, got_v = sol.pick_kivi_quant_axis(K, V)
        except Exception:
            return {"exact_match": 0.0}
        exp_k = _axis_label(K)
        exp_v = _axis_label(V)
        if (got_k != exp_k) or (got_v != exp_v):
            ok = 0.0
            break
    return {"exact_match": ok}
