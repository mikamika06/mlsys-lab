import numpy as np


def tiled_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """Tiled (flash-style) attention forward.

    Sweep K/V in blocks of at most `block_size` rows (the last block may be
    smaller if `block_size` doesn't evenly divide the sequence length).
    Maintain, per query row, a running max `m` and running normalizer `l`
    (online softmax): whenever a new block raises the running max, rescale
    the accumulated output and normalizer by exp(m_old - m_new) before
    folding in the new block's contribution. The final output must be
    identical (up to floating-point rounding) to dense
    softmax(Q @ K.T / sqrt(d)) @ V, for ANY block_size -- including
    block_size == 1 and block_size >= N.

    Q, K, V: (N, d) arrays. Returns (N, d) float64 array.
    """
    raise NotImplementedError('your code here')


def softmax_stability_probe(scores: np.ndarray):
    """Compare a numerically stable softmax against a naive one.

    scores: 2-D array of raw (possibly very large-magnitude) attention
        scores.

    Compute:
      - stable_out: row-wise softmax using max-subtraction before
        exponentiating (must stay finite and correctly normalized
        regardless of how large `scores` gets).
      - unstable_overflowed: True if the naive softmax -- exp(scores)
        normalized by its row sum, WITHOUT subtracting the row max first
        -- produces any non-finite (inf/nan) value on this input; False
        otherwise.

    Returns (stable_out, unstable_overflowed).
    """
    raise NotImplementedError('your code here')
