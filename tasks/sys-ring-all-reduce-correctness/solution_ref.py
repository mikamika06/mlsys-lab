def ring_all_reduce(buffers: list[list[float]]) -> list[list[float]]:
    """
    Simulate ring all-reduce (reduce-scatter then all-gather) across
    len(buffers) ranks arranged in a ring. Returns every rank's final
    buffer, each equal to the elementwise sum of all input buffers.
    """
    N = len(buffers)
    L = len(buffers[0])
    cs = L // N

    def chunk(buf, idx):
        return buf[idx * cs:(idx + 1) * cs]

    state = [list(b) for b in buffers]

    # reduce-scatter: N-1 rounds, each rank accumulates one chunk from its
    # left neighbor into its own copy of that chunk.
    for step in range(N - 1):
        new_state = [list(s) for s in state]
        for r in range(N):
            src = (r - 1) % N
            idx = (src - step) % N
            c_r = chunk(state[r], idx)
            c_src = chunk(state[src], idx)
            summed = [a + b for a, b in zip(c_r, c_src)]
            new_state[r][idx * cs:(idx + 1) * cs] = summed
        state = new_state

    # after reduce-scatter, rank r holds the fully-reduced chunk at index (r+1) % N.
    # all-gather: N-1 rounds, propagate each completed chunk around the ring.
    for step in range(N - 1):
        new_state = [list(s) for s in state]
        for r in range(N):
            src = (r - 1) % N
            idx = (src + 1 - step) % N
            new_state[r][idx * cs:(idx + 1) * cs] = chunk(state[src], idx)
        state = new_state

    return state
