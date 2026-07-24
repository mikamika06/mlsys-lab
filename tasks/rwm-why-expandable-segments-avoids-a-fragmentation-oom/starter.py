def replay_trace(trace, capacity, expandable=False):
    """Simulate a fixed-segment vs. expandable-segment caching allocator.

    Args:
        trace: list of ("alloc", name, size) or ("free", name) ops.
        capacity: max total reserved bytes across all segments.
        expandable: False = legacy fixed segments (no cross-segment
            coalescing); True = one expandable arena (full coalescing).

    Returns:
        {"oom": bool, "peak_reserved": int}
    """
    raise NotImplementedError('your code here')
