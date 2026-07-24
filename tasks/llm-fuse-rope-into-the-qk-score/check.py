import numpy as np
from mlsys.scorers import max_abs_err

def _reference(Q, K, sin, cos):
    # Explicit rotation of Q and K
    Q_even = Q[..., ::2]
    Q_odd  = Q[..., 1::2]
    K_even = K[..., ::2]
    K_odd  = K[..., 1::2]

    Q_rot_even = Q_even * cos - Q_odd * sin
    Q_rot_odd  = Q_even * sin + Q_odd * cos
    K_rot_even = K_even * cos - K_odd * sin
    K_rot_odd  = K_even * sin + K_odd * cos

    Q_rot = np.concatenate([Q_rot_even, Q_rot_odd], axis=-1)
    K_rot = np.concatenate([K_rot_even, K_rot_odd], axis=-1)

    # Compute scores: (batch, seq_len, dim) @ (batch, seq_len, dim).T
    return np.einsum('bld,bmd->blm', Q_rot, K_rot)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    test_cases = [
        (1, 4, 6),
        (3, 7, 10),
        (2, 5, 12),
        (1, 8, 16),
    ]
    max_err = 0.0
    for batch, seq_len, dim in test_cases:
        Q = rng.standard_normal((batch, seq_len, dim)).astype(np.float32)
        K = rng.standard_normal((batch, seq_len, dim)).astype(np.float32)

        half = dim // 2
        angles = np.linspace(0.01, 3.14, seq_len * half).reshape(seq_len, half)
        sin = np.sin(angles).astype(np.float32)
        cos = np.cos(angles).astype(np.float32)

        try:
            cand = sol.fused_rope_qk(Q, K, sin, cos)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _reference(Q, K, sin, cos)

        if cand.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = max_abs_err(ref, cand)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
