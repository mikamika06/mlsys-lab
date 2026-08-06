import numpy as np

def _reference(q_np, k_np, pos_q, pos_k):
    d = q_np.shape[0]
    if d % 2 != 0:
        raise ValueError("q and k must have even length")
    inv_freq = 1 / (10000 ** (np.arange(0, d, 2) / d))
    pos_q_vec = pos_q * inv_freq
    pos_k_vec = pos_k * inv_freq
    sin_q = np.sin(pos_q_vec)
    cos_q = np.cos(pos_q_vec)
    sin_k = np.sin(pos_k_vec)
    cos_k = np.cos(pos_k_vec)

    q_rot = np.empty_like(q_np)
    q_rot[0::2] = q_np[0::2] * cos_q - q_np[1::2] * sin_q
    q_rot[1::2] = q_np[0::2] * sin_q + q_np[1::2] * cos_q

    k_rot = np.empty_like(k_np)
    k_rot[0::2] = k_np[0::2] * cos_k - k_np[1::2] * sin_k
    k_rot[1::2] = k_np[0::2] * sin_k + k_np[1::2] * cos_k

    return float(np.dot(q_rot, k_rot))

def grade(sol, fx) -> dict:
    try:
        func = getattr(sol, "rope_relative_dot")
    except AttributeError:
        return {"rel_err": float("inf")}

    max_rel_err = 0.0
    rng = np.random.default_rng(42)
    for _ in range(10):
        d = 64
        q_arr = rng.standard_normal(d).astype(np.float64)
        k_arr = rng.standard_normal(d).astype(np.float64)
        q = q_arr.tolist()
        k = k_arr.tolist()
        pos_q = int(rng.integers(-20, 21))
        pos_k = int(rng.integers(-20, 21))
        try:
            got = func(q, k, pos_q, pos_k)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(q_arr, k_arr, pos_q, pos_k)
        rel_err = abs(got - ref) / max(abs(ref), 1e-12)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": max_rel_err}
