def grade(sol, fx) -> dict:
    import numpy as np

    rng = np.random.default_rng(42)
    rel_errs = []
    shape_ok = True

    for _ in range(5):
        n = rng.integers(2, 10)
        d = rng.integers(3, 8)
        S = rng.standard_normal((n, d))

        # Reference computation
        m_ref = np.max(S, axis=1)
        l_ref = np.sum(np.exp(S - m_ref[:, None]), axis=1)

        try:
            m_out, l_out = sol.online_softmax_stats(S)
        except Exception:
            return {"rel_err": float("inf"), "shape_ok": 0}

        # Shape check
        if m_out.shape != (n,) or l_out.shape != (n,):
            shape_ok = False

        ref = np.concatenate([m_ref, l_ref])
        out = np.concatenate([m_out, l_out])
        rel_errs.append(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-12))

    return {"rel_err": float(max(rel_errs)), "shape_ok": 1 if shape_ok else 0}
