def compute_roofline_metrics(op, **dims):
    """Return {'flops', 'bytes', 'ai'} for SAXPY or GEMM (FP64)."""
    BYTES_PER_ELEM = 8

    if op == "saxpy":
        n = dims["n"]
        flops = 2 * n
        byte_count = 3 * n * BYTES_PER_ELEM
    elif op == "gemm":
        m, k, n = dims["m"], dims["k"], dims["n"]
        flops = 2 * m * n * k
        byte_count = (m * k + k * n + m * n) * BYTES_PER_ELEM
    else:
        raise ValueError(f"Unknown op: {op}")

    return {
        "flops": float(flops),
        "bytes": float(byte_count),
        "ai": float(flops / byte_count),
    }
