def compute_peak_flops_ratio(frequency_ghz: float) -> float:
    amx_flops_per_cycle = 2048
    avx512_flops_per_cycle = 512
    return float(amx_flops_per_cycle / avx512_flops_per_cycle)
