import numpy as np

from mlsys import scorers


def _oracle(X, W):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    sx = np.max(np.abs(X), axis=1) / 127.0     # (n,)
    sw = np.max(np.abs(W), axis=0) / 127.0     # (m,)

    safe_sx = np.where(sx == 0.0, 1.0, sx)
    safe_sw = np.where(sw == 0.0, 1.0, sw)

    Xq = np.clip(np.round(X / safe_sx[:, None]), -127, 127)
    Wq = np.clip(np.round(W / safe_sw[None, :]), -127, 127)

    acc = Xq.astype(np.int64) @ Wq.astype(np.int64)
    Y = acc.astype(np.float64) * np.outer(sx, sw)
    return Y


def grade(sol, fx) -> dict:
    X = fx["int8_x"]
    W = fx["int8_w"]
    ref = _oracle(X, W)

    try:
        got = np.asarray(sol.vector_wise_int8_matmul(X.copy(), W.copy()), dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf")}

    if got.shape != ref.shape:
        return {"rel_err": float("inf")}

    return {"rel_err": scorers.rel_err(ref, got)}
