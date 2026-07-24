import numpy as np


def tiled_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """
    Tiled (flash-style) attention forward using the online-softmax
    recurrence: sweep key/value blocks of at most `block_size` rows,
    maintaining a running row max `m` and running normalizer `l`, rescaling
    the accumulator whenever the running max increases. This is exact
    (identical to the dense computation, up to float rounding) regardless
    of `block_size`, including when `block_size` doesn't evenly divide the
    sequence length (ragged last block).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    O = np.zeros((N, d), dtype=np.float64)
    m = np.full(N, -np.inf)
    l = np.zeros(N)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Kb = K[start:end]
        Vb = V[start:end]

        S = (Q @ Kb.T) * scale               # (N, block)
        block_max = S.max(axis=1)
        m_new = np.maximum(m, block_max)

        alpha = np.exp(m - m_new)
        alpha = np.where(np.isneginf(m), 0.0, alpha)  # first block: nothing to rescale

        P = np.exp(S - m_new[:, None])
        l = l * alpha + P.sum(axis=1)
        O = O * alpha[:, None] + P @ Vb
        m = m_new

    return O / l[:, None]


def softmax_stability_probe(scores: np.ndarray):
    """
    Compute the numerically stable (max-subtracted) row-wise softmax of
    `scores`, and separately check whether the naive (no max-subtraction)
    softmax would overflow to a non-finite value on this same input.

    Returns (stable_out, unstable_overflowed).
    """
    scores = np.asarray(scores, dtype=np.float64)

    shifted = scores - scores.max(axis=-1, keepdims=True)
    stable_w = np.exp(shifted)
    stable_out = stable_w / stable_w.sum(axis=-1, keepdims=True)

    with np.errstate(over="ignore", invalid="ignore"):
        unstable_w = np.exp(scores)
        unstable_out = unstable_w / unstable_w.sum(axis=-1, keepdims=True)
    unstable_overflowed = bool(not np.all(np.isfinite(unstable_out)))

    return stable_out, unstable_overflowed
