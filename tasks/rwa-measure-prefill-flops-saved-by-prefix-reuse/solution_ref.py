def _cost(L: list[float]) -> list[float]:
    n = len(L)
    out = [0.0] * n
    for i in range(n):
        out[i] = L[i] * (L[i] + 1.0) / 2.0
    return out


def prefill_flops_saved_fraction(
    lengths: list[float], reused_prefix: list[float]
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
    full_cost = _cost(lengths)
    prefix_cost = _cost(reused_prefix)

    n = len(lengths)
    reuse_cost = [0.0] * n
    for i in range(n):
        reuse_cost[i] = full_cost[i] - prefix_cost[i]

    total_full = 0.0
    for i in range(n):
        total_full += full_cost[i]

    total_reuse = 0.0
    for i in range(n):
        total_reuse += reuse_cost[i]

    return 1.0 - total_reuse / total_full
