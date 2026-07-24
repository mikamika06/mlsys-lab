import numpy as np


def _hadamard(n):
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _quantize_int4(x):
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** (4 - 1) - 1
    scale = float(np.max(np.abs(x)))
    scale = scale / qmax if scale > 0 else 1.0
    code = np.clip(np.round(x / scale), -qmax, qmax)
    return code * scale


def _oracle(X, W):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    d = X.shape[1]
    H = _hadamard(d)

    ref = X @ W
    Xr = X @ H.T
    Wr = H @ W
    out_rotated = Xr @ Wr

    Xq = _quantize_int4(X)
    Wq = _quantize_int4(W)
    mse_unrotated = float(np.mean((ref - Xq @ Wq) ** 2))

    Xrq = _quantize_int4(Xr)
    Wrq = _quantize_int4(Wr)
    mse_rotated = float(np.mean((ref - Xrq @ Wrq) ** 2))

    return ref, out_rotated, mse_unrotated, mse_rotated


def _make_case(seed, d, n, m, n_out, mult):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    outlier_idx = rng.choice(d, size=n_out, replace=False)
    X[:, outlier_idx] *= mult
    W = rng.normal(size=(d, m)) * 0.3
    return X, W


def _cases():
    return [
        _make_case(10, 64, 24, 10, 1, 80.0),
        _make_case(11, 32, 16, 8, 1, 60.0),
        _make_case(12, 128, 20, 6, 2, 100.0),
        _make_case(14, 64, 30, 12, 1, 200.0),
    ]


FAIL = {
    "invariance_rel_err": float("inf"),
    "mse_rel_err": float("inf"),
    "rotation_helps": 0.0,
}


def grade(sol, fx) -> dict:
    worst_inv = 0.0
    worst_mse_rel = 0.0
    rotation_helps = 1.0

    for X, W in _cases():
        ref, ref_rot, ref_mse_un, ref_mse_rot = _oracle(X, W)

        try:
            out_rot, mse_un, mse_rot = sol.rotate_and_quantize_matmul(
                np.array(X, copy=True), np.array(W, copy=True)
            )
            out_rot = np.asarray(out_rot, dtype=np.float64)
        except Exception:
            return dict(FAIL)

        if out_rot.shape != ref.shape:
            return dict(FAIL)
        if not np.all(np.isfinite(out_rot)):
            return dict(FAIL)

        inv_err = np.linalg.norm(out_rot - ref) / (np.linalg.norm(ref) + 1e-12)
        worst_inv = max(worst_inv, float(inv_err))

        try:
            mse_un = float(mse_un)
            mse_rot = float(mse_rot)
        except (TypeError, ValueError):
            return dict(FAIL)

        denom = abs(ref_mse_un) + abs(ref_mse_rot) + 1e-12
        mse_rel = (abs(mse_un - ref_mse_un) + abs(mse_rot - ref_mse_rot)) / denom
        worst_mse_rel = max(worst_mse_rel, float(mse_rel))

        if not (mse_rot < mse_un):
            rotation_helps = 0.0

    return {
        "invariance_rel_err": worst_inv,
        "mse_rel_err": worst_mse_rel,
        "rotation_helps": rotation_helps,
    }
