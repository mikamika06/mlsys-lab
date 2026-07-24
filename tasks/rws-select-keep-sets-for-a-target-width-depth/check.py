import numpy as np

def _oracle(width_importance, layer_importance, d_target, L_target):
    w_idx = np.argsort(-width_importance)[:d_target]
    l_idx = np.argsort(-layer_importance)[:L_target]
    return w_idx.astype(np.int64), l_idx.astype(np.int64)

def grade(sol, fx) -> dict:
    # Example test cases
    tests = [
        (np.array([0.2, 0.5, 0.1, 0.9]), np.array([0.3, 0.4, 0.8]), 2, 1),
        (np.arange(10, dtype=float), np.linspace(0, 1, 6), 5, 3),
        (np.random.rand(7), np.random.rand(4), 3, 2),
    ]
    ok = 1.0
    for w_imp, l_imp, d_tgt, L_tgt in tests:
        try:
            got_w, got_l = sol.select_keep_sets(w_imp, l_imp, d_tgt, L_tgt)
        except Exception:
            return {"exact_match": 0.0}
        ref_w, ref_l = _oracle(w_imp, l_imp, d_tgt, L_tgt)
        if not (np.array_equal(got_w, ref_w) and np.array_equal(got_l, ref_l)):
            ok = 0.0
            break
    return {"exact_match": ok}
