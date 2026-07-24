def inclusive_scan_warp(t, n):
    """Inclusive prefix sum within each 32-lane warp, in place, via shuffles.

    `t` is an arena.cuda_sim thread. Because a shuffle is warp-synchronous,
    this must be a generator: `up = yield t.shfl_up(value, delta)`.
    Shared memory is not allowed - the gate requires smem_waves == 0.
    """
    raise NotImplementedError("your code here")
    yield  # keeps this a generator kernel
