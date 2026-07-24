import numpy as np


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked softmax over the kept keys of each query row.

    ``scores`` (n, m) logits, ``mask`` (n, m) bool with True = keep the key.

    BUG: this stable softmax is correct for rows that keep at least one key,
    but a fully padded query (a row whose mask is all False) becomes all
    -inf, so row_max = -inf, the shift -inf - (-inf) = NaN, and the row
    normalises to 0/0 = NaN. Make every row finite: a fully masked row must
    return the all-zeros vector.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    neg = np.where(mask, scores, -np.inf)
    row_max = np.max(neg, axis=-1, keepdims=True)
    e = np.exp(neg - row_max)
    return e / e.sum(axis=-1, keepdims=True)
