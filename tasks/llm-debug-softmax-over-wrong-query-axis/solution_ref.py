import math
import numpy as np

def sdpa(query: np.ndarray,
         key: np.ndarray,
         value: np.ndarray,
         scale: float | None = None) -> np.ndarray:
    """
    Scaled dot‑product attention with correct softmax axis.

    Parameters
    ----------
    query : np.ndarray
        Shape (B, N_q, d_k)
    key : np.ndarray
        Shape (B, N_k, d_k)
    value : np.ndarray
        Shape (B, N_k, d_v)
    scale : float | None, optional
        Scaling factor. If None, defaults to 1/sqrt(d_k).

    Returns
    -------
    np.ndarray
        Attention output of shape (B, N_q, d_v).
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("All inputs must be 3‑D arrays.")
    B, Nq, dk = query.shape
    _, Nk, _ = key.shape
    _, _, dv = value.shape

    if key.shape[2] != dk or value.shape[1] != Nk:
        raise ValueError("Incompatible shapes.")

    if scale is None:
        scale = 1.0 / math.sqrt(dk)

    output = np.empty((B, Nq, dv), dtype=query.dtype)

    for b in range(B):
        for i in range(Nq):
            scores = []
            max_score = -float('inf')
            for j in range(Nk):
                score = 0.0
                for k in range(dk):
                    score += query[b, i, k] * key[b, j, k]
                score *= scale
                scores.append(score)
                if score > max_score:
                    max_score = score

            exp_sum = 0.0
            exp_scores = []
            for j in range(Nk):
                e = math.exp(scores[j] - max_score)
                exp_scores.append(e)
                exp_sum += e

            for d in range(dv):
                out_val = 0.0
                for j in range(Nk):
                    prob = exp_scores[j] / exp_sum
                    out_val += prob * value[b, j, d]
                output[b, i, d] = out_val

    return output
