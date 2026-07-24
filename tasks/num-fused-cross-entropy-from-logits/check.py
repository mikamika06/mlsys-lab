import numpy as np

from mlsys import scorers


def _oracle(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    m = np.max(logits, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.sum(np.exp(logits - m), axis=1))
    tgt_logit = logits[np.arange(logits.shape[0]), targets]
    return lse - tgt_logit


def grade(sol, fx) -> dict:
    logits = np.asarray(fx["logits"], dtype=np.float64)
    targets = np.asarray(fx["targets"], dtype=np.int64)

    ref = _oracle(logits, targets)

    try:
        got = np.asarray(sol.fused_cross_entropy(logits, targets), dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf")}

    if got.shape != ref.shape or not np.all(np.isfinite(got)):
        return {"rel_err": float("inf")}

    return {"rel_err": scorers.rel_err(ref, got)}
