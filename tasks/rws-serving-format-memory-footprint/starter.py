def sparse_2_4_footprint(d_out: int, d_in: int, use_int4: bool):
    """
    Compute the served byte footprint of a (d_out, d_in) weight stored in
    2:4 structured-sparse format (2 nonzeros kept per contiguous group of
    4 columns), optionally with int4-packed kept values, as described in
    task.md. `d_in` is guaranteed divisible by 4.

    Returns (total_bytes, size_ratio) where size_ratio is
    (dense fp16 bytes) / total_bytes.
    """
    raise NotImplementedError('your code here')
