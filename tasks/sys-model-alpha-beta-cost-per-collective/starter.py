def collective_cost(collective: str, num_ranks: int, message_bytes: float,
                     alpha: float, beta: float) -> dict:
    """Standard alpha-beta cost model for a collective communication
    operation. See task.md for the exact per-collective algorithm and
    formulas.

    Returns {"latency_term": ..., "bandwidth_term": ..., "total": ...}.
    """
    raise NotImplementedError('your code here')
