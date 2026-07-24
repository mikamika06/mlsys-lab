import numpy as np


def _reference(H, N_h, P, dtype_bytes, peak_flops_per_s, peak_bytes_per_s):
    del N_h  # cancels out of every closed-form expression below
    weight_bytes = 12 * H * H * dtype_bytes
    S = P

    flops_pre = 12 * H * H * P + 2 * H * P * P
    bytes_pre = weight_bytes + 3 * dtype_bytes * P * H
    ai_pre = flops_pre / bytes_pre

    flops_dec = 12 * H * H + 2 * H * S
    bytes_dec = weight_bytes + dtype_bytes * (2 * S * H + 3 * H)
    ai_dec = flops_dec / bytes_dec

    ridge = peak_flops_per_s / peak_bytes_per_s
    cls_pre = "compute-bound" if ai_pre >= ridge else "bandwidth-bound"
    cls_dec = "compute-bound" if ai_dec >= ridge else "bandwidth-bound"

    return {
        "prefill": {"flops": flops_pre, "bytes": bytes_pre, "ai": ai_pre, "roofline_class": cls_pre},
        "decode": {"flops": flops_dec, "bytes": bytes_dec, "ai": ai_dec, "roofline_class": cls_dec},
    }


def grade(sol, fx) -> dict:
    """Builds random (H, N_h, P, dtype_bytes, hardware-profile) configs and
    compares the candidate's returned FLOPs/bytes/AI against an
    independently computed closed-form reference (`modeled_arith_intensity`
    = max relative error over all 6 numeric fields), and its
    "compute-bound"/"bandwidth-bound" labels against the reference
    classification derived from the *reference* AI and the given hardware
    ridge point (`classification_exact`).
    """
    rng = np.random.default_rng(0)

    max_rel_err = 0.0
    all_classes_ok = 1.0

    for _ in range(6):
        H = int(rng.choice([512, 768, 1024, 2048, 4096]))
        N_h = int(rng.choice([8, 16, 32]))
        P = int(rng.integers(64, 4096))
        dtype_bytes = int(rng.choice([2, 4]))
        peak_flops_per_s = float(rng.uniform(5e12, 50e12))
        peak_bytes_per_s = float(rng.uniform(100e9, 900e9))

        ref = _reference(H, N_h, P, dtype_bytes, peak_flops_per_s, peak_bytes_per_s)

        try:
            got = sol.prefill_decode_roofline(
                H, N_h, P, dtype_bytes, peak_flops_per_s, peak_bytes_per_s
            )
        except Exception:
            return {"modeled_arith_intensity": float("inf"), "classification_exact": 0.0}

        try:
            for phase in ("prefill", "decode"):
                for field in ("flops", "bytes", "ai"):
                    r = float(ref[phase][field])
                    g = float(got[phase][field])
                    max_rel_err = max(max_rel_err, abs(g - r) / (abs(r) + 1e-12))
                if got[phase]["roofline_class"] != ref[phase]["roofline_class"]:
                    all_classes_ok = 0.0
        except Exception:
            return {"modeled_arith_intensity": float("inf"), "classification_exact": 0.0}

    return {
        "modeled_arith_intensity": float(max_rel_err),
        "classification_exact": float(all_classes_ok),
    }
