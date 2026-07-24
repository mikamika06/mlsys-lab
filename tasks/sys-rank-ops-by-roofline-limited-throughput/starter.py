def rank_kernels_by_throughput(kernels: list[tuple[float, float]], peak_flops: float, peak_bw: float) -> list[int]:
    """
    kernels: list of (flops, bytes_moved) pairs.
    peak_flops: machine peak compute rate (FLOP/s).
    peak_bw: machine peak memory bandwidth (bytes/s).

    Returns the kernel indices sorted by attainable roofline throughput
    min(peak_flops, (flops/bytes_moved) * peak_bw), highest first,
    ties broken by ascending original index.
    """
    raise NotImplementedError('your code here')
