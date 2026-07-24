import numpy as np

from mlsys import scorers


def _oracle(X, W, V, scale, bits):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1

    r = V / scale[:, None]
    mask = (np.abs(r) <= qmax + 0.5).astype(np.float64)
    codes = np.clip(np.round(r), -qmax, qmax)
    Wq = scale[:, None] * codes

    B, O = X.shape[0], W.shape[0]
    pred = X @ Wq.T
    target = X @ W.T
    diff = pred - target

    grad = mask * (2.0 / (B * O)) * (diff.T @ X)
    return grad


def _build_cases():
    cases = []
    for seed, B, O, I, bits in [(0, 12, 6, 10, 4), (1, 8, 4, 16, 3), (2, 20, 10, 8, 4)]:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((B, I))
        W = rng.standard_normal((O, I))
        V = W + rng.standard_normal((O, I)) * 0.3
        scale = np.abs(rng.standard_normal(O)) * 0.5 + 0.2
        cases.append((X, W, V, scale, bits))
    return cases


def grade(sol, fx) -> dict:
    all_ref = []
    all_got = []
    for X, W, V, scale, bits in _build_cases():
        ref = _oracle(X, W, V, scale, bits)

        try:
            got = np.asarray(sol.ste_block_mse_grad_wrt_v(X.copy(), W.copy(), V.copy(), scale.copy(), bits),
                              dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"rel_err": float("inf")}

        all_ref.append(ref.reshape(-1))
        all_got.append(got.reshape(-1))

    return {"rel_err": scorers.rel_err(np.concatenate(all_ref), np.concatenate(all_got))}
