"""Milestone 2 checker: NUMA access ratios and locality efficiency."""

import ref


def check(workdir):
    from scaling.numa import calculate_numa_ratios, evaluate_locality_efficiency

    out = {"numa_ratio_matched": 0.0, "local_ratio_matched": 0.0}

    ref_numa_ratio = ref.calculate_numa_ratios(ref.DISTANCE_MATRIX)
    got_numa_ratio = calculate_numa_ratios(ref.DISTANCE_MATRIX)

    if abs(ref_numa_ratio - got_numa_ratio) < 1e-4:
        out["numa_ratio_matched"] = 1.0
    else:
        out["_note"] = f"numa ratio: expected {ref_numa_ratio}, got {got_numa_ratio}"

    ref_eff = ref.evaluate_locality_efficiency(ref.ACCESS_LOG, ref.DISTANCE_MATRIX)
    got_eff = evaluate_locality_efficiency(ref.ACCESS_LOG, ref.DISTANCE_MATRIX)

    if abs(ref_eff - got_eff) < 1e-4:
        out["local_ratio_matched"] = 1.0
    else:
        if "_note" not in out:
            out["_note"] = f"locality efficiency: expected {ref_eff}, got {got_eff}"

    return out
