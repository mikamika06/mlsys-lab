import numpy as np


def ring_all_reduce(buffers: list[np.ndarray]) -> list[np.ndarray]:
    """
    Simulate ring all-reduce (reduce-scatter then all-gather) across
    len(buffers) ranks arranged in a ring. Returns every rank's final
    buffer, each equal to the elementwise sum of all input buffers.
    """
    N = len(buffers)
    L = buffers[0].shape[0]
    cs = L // N

    def chunk(buf, idx):
        return buf[idx * cs:(idx + 1) * cs]

    state = [b.astype(np.float64).copy() for b in buffers]

    # reduce-scatter: N-1 rounds, each rank accumulates one chunk from its
    # left neighbor into its own copy of that chunk.
    for step in range(N - 1):
        new_state = [s.copy() for s in state]
        for r in range(N):
            src = (r - 1) % N
            idx = (src - step) % N
            new_state[r][idx * cs:(idx + 1) * cs] = chunk(state[r], idx) + chunk(state[src], idx)
        state = new_state

    # after reduce-scatter, rank r holds the fully-reduced chunk at index (r+1) % N.
    # all-gather: N-1 rounds, propagate each completed chunk around the ring.
    for step in range(N - 1):
        new_state = [s.copy() for s in state]
        for r in range(N):
            src = (r - 1) % N
            idx = (src + 1 - step) % N
            new_state[r][idx * cs:(idx + 1) * cs] = chunk(state[src], idx)
        state = new_state

    return state
