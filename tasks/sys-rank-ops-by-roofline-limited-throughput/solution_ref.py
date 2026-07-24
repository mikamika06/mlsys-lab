def rank_kernels_by_throughput(kernels: list[tuple[float, float]], peak_flops: float, peak_bw: float) -> list[int]:
    """
    kernels: list of (flops, bytes_moved) pairs.
    peak_flops: machine peak compute rate (FLOP/s).
    peak_bw: machine peak memory bandwidth (bytes/s).

    Returns the kernel indices sorted by attainable roofline throughput
    min(peak_flops, (flops/bytes_moved) * peak_bw), highest first,
    ties broken by ascending original index.
    """
    attainable = []
    for flops, bytes_moved in kernels:
        ai = flops / bytes_moved
        attainable.append(min(peak_flops, ai * peak_bw))

    order = sorted(range(len(kernels)), key=lambda i: (-attainable[i], i))
    return order
