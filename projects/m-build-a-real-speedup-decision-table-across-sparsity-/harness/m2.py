import ref


def check(workdir):
    from sparsity.hardware import (
        validate_nm_tensorcore_alignment,
        compute_nm_speedup_gap,
    )

    out = {
        "validations_match": 1.0,
        "speedups_match": 1.0,
    }

    for hw in ref.HW_CASES:
        M, N, K = hw["M"], hw["N"], hw["K"]
        bits = hw["dtype_bits"]
        sp = hw["sparsity"]
        bw = hw["bandwidth_gbps"]
        tf = hw["compute_tflops"]

        want_val = ref.ref_validate_nm_alignment(M, N, K, bits)
        got_val = validate_nm_tensorcore_alignment(M, N, K, bits)
        if want_val != got_val:
            out["validations_match"] = 0.0
            out["_note"] = f"Validation mismatch: want {want_val}, got {got_val}"
            return out

        want_gap = ref.ref_compute_nm_speedup_gap(M, N, K, sp, bw, tf, bits)
        got_gap = compute_nm_speedup_gap(M, N, K, sp, bw, tf, bits)

        for key in ["theoretical_speedup", "achievable_speedup", "speedup_gap"]:
            if abs(want_gap[key] - got_gap[key]) > 1e-3:
                out["speedups_match"] = 0.0
                out["_note"] = f"Speedup mismatch on {key}: want {want_gap[key]}, got {got_gap[key]}"
                return out

    return out
