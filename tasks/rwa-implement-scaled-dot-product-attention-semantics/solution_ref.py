import numpy as np
import math
import itertools

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    causal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot‑product attention with optional masking.

    Parameters
    ----------
    Q : (..., T_q, d_k)
        Query tensor.
    K : (..., T_k, d_k)
        Key tensor.
    V : (..., T_k, d_v)
        Value tensor.
    mask : array_like or None, optional
        Broadcastable to the logits shape. If bool, positions with False are masked out.
        If float, values are added element‑wise to the logits before softmax.
    causal : bool, default False
        Whether to apply a causal (triangular) mask.

    Returns
    -------
    output : (..., T_q, d_v)
        Aggregated values.
    weights : (..., T_q, T_k)
        Attention probabilities.
    """
    d_k = K.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    batch_shape = Q.shape[:-2]
    T_q = Q.shape[-2]
    T_k = K.shape[-2]
    d_v = V.shape[-1]

    output_shape = batch_shape + (T_q, d_v)
    weights_shape = batch_shape + (T_q, T_k)

    output = np.zeros(output_shape, dtype=Q.dtype)
    weights = np.zeros(weights_shape, dtype=Q.dtype)

    batch_indices = list(itertools.product(*(range(s) for s in batch_shape)))
    if not batch_indices:
        batch_indices = [()]

    for idx in batch_indices:
        q_slice = Q[idx]
        k_slice = K[idx]
        v_slice = V[idx]
        mask_slice = mask[idx] if mask is not None else None

        logits_slice = np.zeros((T_q, T_k), dtype=Q.dtype)
        for i in range(T_q):
            for j in range(T_k):
                dot = 0.0
                for l in range(d_k):
                    dot += q_slice[i, l] * k_slice[j, l]
                logits_slice[i, j] = dot * scale

        if causal:
            for i in range(T_q):
                for j in range(T_k):
                    if j > i:
                        logits_slice[i, j] += -float('inf')

        if mask_slice is not None:
            if mask_slice.dtype == bool:
                for i in range(T_q):
                    for j in range(T_k):
                        if not mask_slice[i, j]:
                            logits_slice[i, j] = -float('inf')
            else:
                for i in range(T_q):
                    for j in range(T_k):
                        logits_slice[i, j] += mask_slice[i, j]

        weights_slice = np.zeros((T_q, T_k), dtype=Q.dtype)
        for i in range(T_q):
            max_val = -float('inf')
            for j in range(T_k):
                val = logits_slice[i, j]
                if val > max_val:
                    max_val = val

            row_exp = []
            row_sum = 0.0
            for j in range(T_k):
                e = math.exp(logits_slice[i, j] - max_val)
                row_exp.append(e)
                row_sum += e

            for j in range(T_k):
                weights_slice[i, j] = row_exp[j] / row_sum

        output_slice = np.zeros((T_q, d_v), dtype=Q.dtype)
        for i in range(T_q):
            for c in range(d_v):
                acc = 0.0
                for j in range(T_k):
                    acc += weights_slice[i, j] * v_slice[j, c]
                output_slice[i, c] = acc

        output[idx] = output_slice
        weights[idx] = weights_slice

    return output, weights
