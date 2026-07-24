import numpy as np

def _reference(teacher_logits, student_logits):
    t = teacher_logits.ravel()
    s = student_logits.ravel()
    if len(t) < len(s):
        t = np.concatenate([t, np.zeros(len(s)-len(t))])
    elif len(s) < len(t):
        s = np.concatenate([s, np.zeros(len(t)-len(s))])
    t_sorted = np.sort(t)
    s_sorted = np.sort(s)
    return float(np.sum(np.abs(t_sorted - s_sorted)))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_rel_err = 0.0
    for _ in range(10):
        len_t = rng.integers(1, 51)
        len_s = rng.integers(1, 51)
        t = rng.standard_normal(len_t)
        s = rng.standard_normal(len_s)
        ref = _reference(t, s)
        try:
            cand = sol.wasserstein_1_loss_on_sorted_logits(t, s)
        except Exception:
            return {"rel_err": float("inf")}
        rel_err = abs(cand - ref) / (abs(ref) + 1e-12)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": max_rel_err}
