def simulate(trace, l1_cap, l2_cap, policy="always", write_mode="wb"):
    """
    Simulates a 3-tier exclusive cache.
    trace: list of (op, key, size) where op is 'R' or 'W'.
    policy: 'always', 'reuse_2', 'size_aware'
    write_mode: 'wb', 'wt'
    Returns dict: {'latency_ns': int, 'write_penalty_ns': int, 'l1_evictions': int, 'l2_evictions': int}
    """
    raise NotImplementedError
