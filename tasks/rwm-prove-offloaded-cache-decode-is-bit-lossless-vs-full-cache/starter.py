import math

def offloaded_decode_attention(Q: list[list[list[list[float]]]], K_new: list[list[list[list[float]]]], V_new: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    """Multi-layer, multi-head causal decode attention over a KV cache that
    lives in a CPU-resident "offload store" and is gathered back per step.

    Parameters
    ----------
    Q, K_new, V_new : list[float], shape (L, T, H, d)
        L layers, T decode steps, H heads, d head dim.

    Returns
    -------
    out : list[float], shape (L, T, H, d)

    BUG: at each decode step this only gathers the pair that was JUST
    pushed to the offload store, not the full history accumulated so far --
    so every step attends only to itself instead of every prior offloaded
    token. Fix the gather.
    """
    raise NotImplementedError('your code here')
