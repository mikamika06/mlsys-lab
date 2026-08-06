def compute_instruction_counts(vector_len: int) -> dict:
    scalar_instrs = vector_len * 4 + 2
    avx2_vectors = (vector_len + 31) // 32
    avx2_instrs = avx2_vectors * 3 + 2
    vnni_vectors = (vector_len + 63) // 64
    vnni_instrs = vnni_vectors * 1 + 2

    return {
        "vector_len": vector_len,
        "scalar": scalar_instrs,
        "avx2": avx2_instrs,
        "vnni": vnni_instrs,
    }


def compute_theoretical_mac_per_cycle(fma_units: int) -> dict:
    return {
        "fma_units": fma_units,
        "avx2_mac_per_cycle": fma_units * 32,
        "vnni_mac_per_cycle": fma_units * 128,
        "speedup_ratio": 4.0,
    }
