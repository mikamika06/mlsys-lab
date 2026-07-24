def crossover_seq_len(
    recompute_coeff: float,
    kv_bytes_per_token: float,
    bandwidth_bytes_per_s: float,
) -> int:
    transfer_coeff = kv_bytes_per_token / bandwidth_bytes_per_s
    s = transfer_coeff / recompute_coeff
    return max(1, int(__import__("math").ceil(s)))
