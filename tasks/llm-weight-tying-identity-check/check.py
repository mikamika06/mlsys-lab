import numpy as np
from mlsys import scorers


def _oracle_pipeline(E, token_ids):
    """Independent reference: explicit one-hot lookup, then the tied LM head.

    logits = one_hot(token_ids) @ E @ E.T  ==  (E E^T) gathered at token_ids.
    Computed here via the literal one-hot matmul so the check does not merely
    mirror the intended one-line gather.
    """
    E = np.asarray(E, dtype=np.float64)
    V = E.shape[0]
    one_hot = np.eye(V, dtype=np.float64)[np.asarray(token_ids)]  # (n, V)
    hidden = one_hot @ E                                          # (n, d)
    return hidden @ E.T                                           # (n, V)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    # (V, d) pairs, all with d != V so a missing transpose / missing head
    # produces a shape mismatch rather than a coincidentally-valid result.
    cases = [(5, 3), (8, 4), (6, 11), (12, 7), (20, 9)]

    max_err = 0.0
    identity_err = 0.0
    try:
        for V, d in cases:
            E = rng.standard_normal((V, d))
            token_ids = rng.integers(0, V, size=2 * V)

            # Gate 1: full embed -> tied-head pipeline against the oracle.
            ref = _oracle_pipeline(E, token_ids)
            got = np.asarray(sol.tied_identity_logits(E, token_ids), dtype=np.float64)
            if got.shape != ref.shape:
                return {"max_abs_err": float("inf"), "identity_err": float("inf")}
            max_err = max(max_err, scorers.max_abs_err(ref, got))

            # Gate 2: feeding every token id must reconstruct the Gram matrix E E^T.
            all_ids = np.arange(V)
            ref_gram = E @ E.T
            got_gram = np.asarray(sol.tied_identity_logits(E, all_ids), dtype=np.float64)
            if got_gram.shape != ref_gram.shape:
                return {"max_abs_err": float("inf"), "identity_err": float("inf")}
            identity_err = max(identity_err, scorers.max_abs_err(ref_gram, got_gram))
    except Exception:
        return {"max_abs_err": float("inf"), "identity_err": float("inf")}

    return {"max_abs_err": max_err, "identity_err": identity_err}
