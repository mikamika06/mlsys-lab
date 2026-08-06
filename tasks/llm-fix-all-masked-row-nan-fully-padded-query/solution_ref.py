import numpy as np


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked softmax that is safe against fully padded query rows.

    ``scores`` (n, m) logits, ``mask`` (n, m) bool with True = keep the key.
    Rows with at least one kept key are the softmax over their kept entries;
    rows with no kept key are the all-zeros vector (no NaN / inf).
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    neg = np.where(mask, scores, -np.inf)
    row_max = np.max(neg, axis=-1, keepdims=True)
    # Fully masked rows have row_max = -inf; replace with 0 so the shift
    # stays finite. Their exps are all exp(-inf) = 0 regardless.
    safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
    e = np.exp(neg - safe_max)                       # masked positions -> 0
    denom = e.sum(axis=-1, keepdims=True)            # 0 for fully masked rows
    out = np.divide(e, denom, out=np.zeros_like(e), where=denom > 0)
    return out.astype(np.float64)
