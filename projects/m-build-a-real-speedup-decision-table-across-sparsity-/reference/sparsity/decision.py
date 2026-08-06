import numpy as np
from sparsity.storage import compute_theoretical_bytes, compute_csr_breakeven_sparsity
from sparsity.hardware import validate_nm_tensorcore_alignment, compute_nm_speedup_gap


def build_sparsity_decision_table(layer_configs):
    results = []
    for cfg in layer_configs:
        name = cfg["name"]
        shape = (cfg["M"], cfg["K"])
        M, N, K = cfg["M"], cfg["N"], cfg["K"]
        sparsity = cfg["sparsity"]
        dtype_bits = cfg.get("dtype_bits", 16)
        bw = cfg.get("bandwidth_gbps", 900.0)
        tflops = cfg.get("compute_tflops", 312.0)

        dense_b = compute_theoretical_bytes(shape, dtype_bits, "dense", sparsity)
        csr_b = compute_theoretical_bytes(shape, dtype_bits, "csr", sparsity)
        coo_b = compute_theoretical_bytes(shape, dtype_bits, "coo", sparsity)
        nm_b = compute_theoretical_bytes(shape, dtype_bits, "2:4", sparsity)

        align = validate_nm_tensorcore_alignment(M, N, K, dtype_bits)
        gap = compute_nm_speedup_gap(M, N, K, sparsity, bw, tflops, dtype_bits)
        breakeven = compute_csr_breakeven_sparsity(shape, dtype_bits)

        if abs(sparsity - 0.5) < 1e-5 and align["valid"]:
            recommended = "2:4"
        elif sparsity >= breakeven and sparsity > 0.7:
            recommended = "csr" if csr_b < coo_b else "coo"
        else:
            recommended = "dense"

        results.append({
            "name": name,
            "dense_bytes": dense_b,
            "csr_bytes": csr_b,
            "coo_bytes": coo_b,
            "2:4_bytes": nm_b,
            "recommended_format": recommended,
            "achievable_speedup": gap["achievable_speedup"],
            "theoretical_speedup": gap["theoretical_speedup"],
            "is_hardware_aligned": align["valid"],
        })
    return results
