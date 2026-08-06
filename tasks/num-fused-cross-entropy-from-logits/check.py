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
    logits_np = np.asarray(fx["logits"], dtype=np.float64)
    targets_np = np.asarray(fx["targets"], dtype=np.int64)

    ref = _oracle(logits_np, targets_np)

    logits_list = logits_np.tolist()
    targets_list = targets_np.tolist()

    try:
        got_raw = sol.fused_cross_entropy(logits_list, targets_list)
        got = np.asarray(got_raw, dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf")}

    if got.shape != ref.shape or not np.all(np.isfinite(got)):
        return {"rel_err": float("inf")}

    return {"rel_err": scorers.rel_err(ref, got)}
