import numpy as np

from mlsys import scorers


def _oracle(W):
    w = np.asarray(W, dtype=np.float64).reshape(-1)
    blocks = w.reshape(-1, 64)
    return np.max(np.abs(blocks), axis=1)


def grade(sol, fx) -> dict:
    W = fx["nf4_w"]
    ref = _oracle(W)

    try:
        got = np.asarray(sol.nf4_block_absmax_scales(W.copy()), dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf")}

    if got.shape != ref.shape:
        return {"rel_err": float("inf")}

    return {"rel_err": scorers.rel_err(ref, got)}
