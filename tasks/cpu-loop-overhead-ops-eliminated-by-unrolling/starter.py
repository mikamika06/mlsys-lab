def unroll_overhead(N, U):
    """Return (overhead_ops_saved, byte_address_trace) for an N-iteration loop
    unrolled by factor U. Overhead model: 2 ops (increment + branch) per iter."""
    raise NotImplementedError("your code here")
