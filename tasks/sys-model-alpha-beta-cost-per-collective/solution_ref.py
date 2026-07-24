import math


def collective_cost(collective: str, num_ranks: int, message_bytes: float,
                     alpha: float, beta: float) -> dict:
    """Standard alpha-beta cost model for a collective communication
    operation. See task.md for the exact per-collective algorithm and
    formulas.

    Returns {"latency_term": ..., "bandwidth_term": ..., "total": ...}.
    """
    P = num_ranks
    M = message_bytes
    frac = (P - 1) / P if P > 0 else 0.0

    if collective in ("reduce_scatter", "allgather", "alltoall"):
        # ring algorithm: P-1 pairwise steps, each moving ~M/P bytes/node
        steps = max(P - 1, 0)
        latency_term = steps * alpha
        bandwidth_term = frac * M * beta
    elif collective == "allreduce":
        # ring allreduce == reduce_scatter followed by allgather
        steps = 2 * max(P - 1, 0)
        latency_term = steps * alpha
        bandwidth_term = 2.0 * frac * M * beta
    elif collective == "broadcast":
        # binomial tree: ceil(log2(P)) steps, each moving the full message
        steps = math.ceil(math.log2(P)) if P > 1 else 0
        latency_term = steps * alpha
        bandwidth_term = steps * M * beta
    else:
        raise ValueError(f"unknown collective {collective!r}")

    return {
        "latency_term": float(latency_term),
        "bandwidth_term": float(bandwidth_term),
        "total": float(latency_term + bandwidth_term),
    }
