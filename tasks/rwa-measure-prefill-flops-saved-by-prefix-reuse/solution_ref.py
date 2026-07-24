import numpy as np


def _cost(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    return L * (L + 1) / 2.0


def prefill_flops_saved_fraction(
    lengths: np.ndarray, reused_prefix: np.ndarray
) -> float:
    """
    lengths: (n,) full context length L_i of each request in the batch.
    reused_prefix: (n,) number of leading tokens of request i whose KV is
    already cached from a prior request sharing that prefix
    (0 <= reused_prefix_i <= lengths_i).

    Causal prefill cost model: processing the token at position i costs
    FLOPs proportional to i (it attends over i prior tokens), so a fresh
    request of length L costs sum_{i=1}^{L} i = L(L+1)/2.

    With `reused_prefix_i` tokens already cached, only positions
    reused_prefix_i+1 .. L_i are actually prefilled for request i -- each
    of those still costs its own index i in FLOPs, since it still
    attends across the full context including the reused, cached keys.
    So request i's reuse cost is L_i(L_i+1)/2 - p_i(p_i+1)/2.

    Return the fraction of total batch prefill FLOPs saved by reuse:
    1 - sum(reuse_cost) / sum(full_cost).
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    reused_prefix = np.asarray(reused_prefix, dtype=np.float64)

    full_cost = _cost(lengths)
    reuse_cost = full_cost - _cost(reused_prefix)

    total_full = float(np.sum(full_cost))
    total_reuse = float(np.sum(reuse_cost))
    return 1.0 - total_reuse / total_full
