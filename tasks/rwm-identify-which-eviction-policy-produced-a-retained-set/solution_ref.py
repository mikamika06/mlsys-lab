import collections


def identify_policy(retained_set, stream):
    """
    Determine which eviction policy produced `retained_set` from the given
    token `stream`.  The function recomputes all three policies and returns
    the matching label.
    """

    # Window (LRU) – most recent first
    cache = collections.OrderedDict()
    for t in stream:
        if t in cache:
            del cache[t]
        cache[t] = None
        if len(cache) > 5:          # WINDOW_SIZE
            cache.popitem(last=False)
    window_set = list(reversed(list(cache.keys())))

    # Heavy‑hitter – top K by frequency, ties broken by token value
    freq = collections.Counter(stream)
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    heavy_hitter_set = [t for t, _ in items[:3]]   # HEAVY_HITTER_K

    # Recent‑only (FIFO) – last N tokens preserving order
    dq = collections.deque(maxlen=5)                # RECENT_ONLY_N
    for t in stream:
        dq.append(t)
    recent_only_set = list(dq)

    if retained_set == window_set:
        return "window"
    elif retained_set == heavy_hitter_set:
        return "heavy_hitter"
    elif retained_set == recent_only_set:
        return "recent_only"
    else:
        raise ValueError("Retained set does not match any known policy")
