import numpy as np

def _rope_pi_ref(seq_len, dim, L_train, L_new):
    """Reference: PI applied to positions only."""
    pos = np.arange(seq_len, dtype=np.float64)
    scale = L_train / L_new
    pos_scaled = pos * scale
    k = np.arange(dim // 2, dtype=np.float64)
    theta = 1.0 / (10000.0 ** (2.0 * k / dim))
    angles = np.outer(pos_scaled, theta)
    return np.cos(angles), np.sin(angles)

def grade(sol, fx) -> dict:
    test_cases = [
        (8,  16, 2048, 4096),
        (16, 64, 2048, 8192),
        (32, 32, 1024, 4096),
        (4,  8,  512,  2048),
    ]
    max_err = 0.0
    for seq_len, dim, L_train, L_new in test_cases:
        ref_cos, ref_sin = _rope_pi_ref(seq_len, dim, L_train, L_new)
        try:
            got = sol.rope_pi(seq_len, dim, L_train, L_new)
            got_cos, got_sin = np.asarray(got[0], dtype=np.float64), np.asarray(got[1], dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        err = max(float(np.max(np.abs(got_cos - ref_cos))),
                  float(np.max(np.abs(got_sin - ref_sin))))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err}
