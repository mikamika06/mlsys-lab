def classify_scheduling(active: list[list[int]]) -> str:
    """Classify a per-iteration active-ID trace as "static" or "continuous".

    active: (T, N) list of lists, active[t][i] truthy iff sequence i is active at
    iteration t. Continuous iff some iteration admits a new ID while a
    previous member is still active; static otherwise.
    """
    T = len(active)
    if T == 0:
        return "static"
    N = len(active[0])
    for t in range(1, T):
        has_new = False
        has_continuing = False
        for i in range(N):
            prev_val = bool(active[t - 1][i])
            curr_val = bool(active[t][i])
            new_id = curr_val and (not prev_val)
            continuing = prev_val and curr_val
            if new_id:
                has_new = True
            if continuing:
                has_continuing = True
        if has_new and has_continuing:
            return "continuous"
    return "static"
