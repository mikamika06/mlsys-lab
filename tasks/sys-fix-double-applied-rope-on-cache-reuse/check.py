import numpy as np


def _rope(x, positions, base: float = 10000.0):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    positions = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    d = x.shape[-1]
    i = np.arange(d // 2)
    theta = base ** (-2.0 * i / d)
    angles = positions[:, None] * theta[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _oracle_step(K_raw, Q_raw, V, pos):
    """Full causal attention over prefix [0..pos], each key/query rotated
    exactly once at its own absolute position -- computed directly from the
    raw (un-rotated) tensors, independent of any cache bookkeeping."""
    d = K_raw.shape[1]
    K_rot = _rope(K_raw[: pos + 1], np.arange(pos + 1))
    q_rot = _rope(Q_raw[pos], np.array([pos]))[0]
    scores = (K_rot @ q_rot) / np.sqrt(d)
    w = _softmax(scores)
    return w @ V[: pos + 1]


def _run_sequence(sol, K_raw, Q_raw, V):
    d = K_raw.shape[1]
    cache = {"k": np.zeros((0, d), dtype=np.float64), "v": np.zeros((0, d), dtype=np.float64)}
    worst = 0.0
    for pos in range(K_raw.shape[0]):
        ref = _oracle_step(K_raw, Q_raw, V, pos)
        try:
            out, cache = sol.decode_step(cache, K_raw[pos], V[pos], Q_raw[pos], pos)
            out = np.asarray(out, dtype=np.float64).reshape(-1)
        except Exception:
            return float("inf")
        if out.shape != ref.shape:
            return float("inf")
        err = float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-12))
        worst = max(worst, err)
    return worst


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst = 0.0
    for T, d in [(3, 4), (5, 8), (6, 4), (4, 16)]:
        K_raw = rng.standard_normal((T, d))
        Q_raw = rng.standard_normal((T, d))
        V = rng.standard_normal((T, d))
        worst = max(worst, _run_sequence(sol, K_raw, Q_raw, V))
    return {"rel_err": worst}
