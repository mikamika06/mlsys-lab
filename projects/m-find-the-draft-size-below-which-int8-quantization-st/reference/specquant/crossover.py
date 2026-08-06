"""Crossover draft model size analysis."""

from specquant.throughput import compute_throughput_ratio


def find_int8_crossover_size(candidate_sizes_m, draft_len, system_config, alpha_config):
    """Finds draft model size below which INT8 quantization stops paying off."""
    sorted_sizes = sorted(candidate_sizes_m)
    crossover = None
    for s in sorted_sizes:
        ratio = compute_throughput_ratio(s, draft_len, system_config, alpha_config)
        if ratio >= 1.0:
            crossover = s
            break
    if crossover is None:
        return float(sorted_sizes[-1])
    return float(crossover)
