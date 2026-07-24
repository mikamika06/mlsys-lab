import numpy as np


def _oracle(nbits, group_size, scale_bits, zero_bits):
    nbits = np.asarray(nbits, dtype=np.float64)
    group_size = np.asarray(group_size, dtype=np.float64)
    scale_bits = np.asarray(scale_bits, dtype=np.float64)
    zero_bits = np.asarray(zero_bits, dtype=np.float64)
    effective_bpv = nbits + (scale_bits + zero_bits) / group_size
    compression_ratio = effective_bpv / np.float64(16.0)
    return float(effective_bpv), float(compression_ratio)


def grade(sol, fx) -> dict:
    cases = [
        (4, 128, 16, 0),
        (8, 64, 16, 8),
        (2, 32, 32, 0),
        (3, 256, 16, 16),
        (6, 96, 24, 8),
        (16, 1, 0, 0),
    ]

    ok = 1.0
    for nbits, group_size, scale_bits, zero_bits in cases:
        expected_bpv, expected_ratio = _oracle(
            nbits, group_size, scale_bits, zero_bits
        )
        try:
            got_bpv, got_ratio = sol.effective_bits_per_value(
                nbits, group_size, scale_bits, zero_bits
            )
            got_bpv = float(got_bpv)
            got_ratio = float(got_ratio)
        except Exception:
            ok = 0.0
            break

        bpv_err = abs(got_bpv - expected_bpv) / (abs(expected_bpv) + 1e-12)
        ratio_err = abs(got_ratio - expected_ratio) / (abs(expected_ratio) + 1e-12)
        if bpv_err > 1e-9 or ratio_err > 1e-9:
            ok = 0.0
            break

    return {"size_ratio": ok}
