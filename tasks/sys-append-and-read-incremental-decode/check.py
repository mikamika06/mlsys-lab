import numpy as np

def _reference_incremental(embeddings, Wq, Wk, Wv):
    """
    Compute incremental attention outputs using a full pre‑fill reference.
    """
    n = embeddings.shape[0]
    d_k = Wk.shape[1]
    sqrt_dk = np.sqrt(d_k)
    # Precompute all Q, K, V
    Q = embeddings @ Wq  # (n, d_q)
    K = embeddings @ Wk  # (n, d_k)
    V = embeddings @ Wv  # (n, d_v)

    outputs = np.empty((n, V.shape[1]), dtype=np.float64)
    for t in range(n):
        scores = (Q[t] @ K[:t+1].T) / sqrt_dk          # shape (t+1,)
        # stable softmax
        max_score = scores.max()
        exp_scores = np.exp(scores - max_score)
        alphas = exp_scores / exp_scores.sum()
        outputs[t] = alphas @ V[:t+1]
    return outputs

def grade(sol, fx) -> dict:
    """
    Grade the candidate solution.

    Parameters
    ----------
    sol : module
        The student's implementation module.
    fx : dict
        Dictionary of fixture data (unused here).

    Returns
    -------
    dict
        Mapping from metric name to score.  For this task we provide a single
        metric `max_abs_err`.
    """
    # Generate random test case
    rng = np.random.default_rng(12345)
    n, d_in = 7, 5
    d_k, d_v = 4, 3

    embeddings = rng.standard_normal((n, d_in))
    Wq = rng.standard_normal((d_in, d_k))
    Wk = rng.standard_normal((d_in, d_k))
    Wv = rng.standard_normal((d_in, d_v))

    try:
        got = sol.incremental_decode(embeddings, Wq, Wk, Wv)
        ref = _reference_incremental(embeddings, Wq, Wk, Wv)
    except Exception as e:
        # Any exception results in a huge error
        return {"max_abs_err": float("inf")}

    if got.shape != ref.shape or got.dtype != np.float64:
        return {"max_abs_err": float("inf")}

    err = np.max(np.abs(got - ref))
    return {"max_abs_err": float(err)}
