import numpy as np


def tied_identity_logits(E: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Logits from the tied embed -> LM-head round trip.

    Embedding lookup gathers rows of E (one-hot @ E collapses to a gather); the
    tied head then projects back through E.T. The whole pipeline equals rows of
    the Gram matrix E @ E.T.

    Parameters
    ----------
    E : np.ndarray
        Tied weight matrix, shape (V, d).
    token_ids : np.ndarray
        1-D array of token ids, shape (n,).

    Returns
    -------
    np.ndarray
        Logits of shape (n, V), dtype float64.
    """
    E = np.asarray(E, dtype=np.float64)
    ids = np.asarray(token_ids)
    hidden = E[ids]          # embedding lookup: (n, d)
    return hidden @ E.T      # tied LM head: (n, V)
