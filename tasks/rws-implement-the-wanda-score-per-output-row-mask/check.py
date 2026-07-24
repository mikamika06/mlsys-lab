import numpy as np


def _oracle(W, X, sparsity):
    col_norm = np.linalg.norm(X, axis=0)
    S = np.abs(W) * col_norm[None, :]
    d_out, d_in = W.shape
    k = max(1, int(round((1.0 - sparsity) * d_in)))

    order = np.argsort(-S, axis=1, kind="stable")
    mask = np.zeros((d_out, d_in), dtype=bool)
    rows_idx = np.arange(d_out)[:, None]
    mask[rows_idx, order[:, :k]] = True
    return mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return 1.0 if union == 0 else inter / union


def grade(sol, fx) -> dict:
    W = fx["ww"]
    X = fx["wx"]

    sparsities = [0.25, 0.5, 0.75]
    min_iou = 1.0

    for sparsity in sparsities:
        ref_mask = _oracle(W, X, sparsity)

        try:
            got = np.asarray(sol.wanda_score_mask(W.copy(), X.copy(), sparsity)).astype(bool)
        except Exception:
            return {"iou": 0.0}

        if got.shape != ref_mask.shape:
            return {"iou": 0.0}

        min_iou = min(min_iou, _iou(got, ref_mask))

    return {"iou": min_iou}
