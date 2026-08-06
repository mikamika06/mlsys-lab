def tmul_vs_avx512_ratio(dtype):
    """Return TMUL vs AVX-512 throughput ratio."""
    if dtype in ("bf16", "int8"):
        return 4.0
    raise ValueError(f"unknown dtype {dtype}")
