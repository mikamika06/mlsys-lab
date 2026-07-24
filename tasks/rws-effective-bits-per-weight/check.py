import numpy as np


def _oracle(N, M, bits, group_size, scale_bits):
    density = N / M
    index_bits = density * float(np.ceil(np.log2(M)))
    scale_overhead = scale_bits / group_size
    return density * bits + index_bits + scale_overhead


def _build_cases():
    return [
        (4, 4, 4, 64, 16.0),    # dense baseline, no sparsity
        (2, 4, 4, 64, 16.0),    # classic 2:4
        (1, 4, 4, 128, 16.0),
        (2, 8, 3, 32, 16.0),
        (4, 8, 8, 256, 32.0),
        (1, 8, 2, 64, 8.0),
        (8, 8, 4, 32, 16.0),
    ]


def grade(sol, fx) -> dict:
    worst_rel = 0.0
    for N, M, bits, group_size, scale_bits in _build_cases():
        ref = _oracle(N, M, bits, group_size, scale_bits)
        try:
            got = float(sol.effective_bits_per_weight(N, M, bits, group_size, scale_bits=scale_bits))
        except Exception:
            return {"rel_err": float("inf")}

        if not np.isfinite(got):
            return {"rel_err": float("inf")}

        rel = abs(got - ref) / (abs(ref) + 1e-12)
        worst_rel = max(worst_rel, rel)

    return {"rel_err": worst_rel}
