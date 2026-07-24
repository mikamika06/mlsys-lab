def replay_trace(trace, capacity, max_split_size=None):
    """Simulate a single-arena caching allocator over an alloc/free trace.

    Args:
        trace: list of ("alloc", name, size) or ("free", name) ops.
        capacity: max arena size in bytes.
        max_split_size: None, or an int cap above which a free block may
            only be reused by an exact-size request (never split).

    Returns:
        {"oom": bool, "peak_reserved": int}
    """
    raise NotImplementedError('your code here')
