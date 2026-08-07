def throughput(trace: list[list[int]]) -> list[float]:
    """
    Compute aggregate and single‑stream throughput from a binary occupancy trace.

    Parameters
    ----------
    trace : list[list[int]]
        2‑D list of shape (steps, slots) with entries 0 or 1.

    Returns
    -------
    list[float]
        [aggregate_throughput, single_stream_rate].
    """
    steps = len(trace)
    slots = len(trace[0]) if steps > 0 else 0

    tokens_per_step = [0.0] * steps
    for i in range(steps):
        row_sum = 0.0
        for j in range(slots):
            row_sum += trace[i][j]
        tokens_per_step[i] = row_sum

    agg_sum = 0.0
    single_count = 0.0
    for i in range(steps):
        val = tokens_per_step[i]
        agg_sum += val
        if val > 0:
            single_count += 1.0

    agg = agg_sum / steps if steps > 0 else 0.0
    single = single_count / steps if steps > 0 else 0.0

    return [agg, single]
