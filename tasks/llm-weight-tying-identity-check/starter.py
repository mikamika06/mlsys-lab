import numpy as np


def tied_identity_logits(E: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Logits from the tied embed -> LM-head round trip.

    Given the tied weight matrix E of shape (V, d) and token_ids of shape (n,),
    return the (n, V) float64 logits produced by embedding each token id and then
    applying the tied LM head (h @ E.T). Vectorize with NumPy; do not loop over
    tokens in Python.
    """
    raise NotImplementedError("your code here")
