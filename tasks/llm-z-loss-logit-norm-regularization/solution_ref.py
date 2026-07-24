import numpy as np


def z_loss(logits: np.ndarray, targets: np.ndarray, lambda_: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    m = np.max(logits, axis=1)
    lse = m + np.log(np.sum(np.exp(logits - m[:, None]), axis=1))
    ce = -logits[np.arange(logits.shape[0]), targets] + lse
    return ce + lambda_ * (lse ** 2)
