def grade(sol, fx) -> dict:
    import numpy as np

    # Test cases: (H, k)
    cases = [
        (1024, 0),
        (2048, 256),
        (4096, 512),
        (8192, 1024),
        (12345, 678),
    ]

    errors = []
    for H, k in cases:
        try:
            bytes_, ratio_ = sol.mixed_precision_memory_accounting(H, k)
        except Exception:
            return {"rel_err": 1.0}

        ref_bytes = (H - k) * 1 + k * 2
        ref_ratio = ref_bytes / (2 * H)

        err = abs(ratio_ - ref_ratio) / (ref_ratio + 1e-12)
        errors.append(err)

    return {"rel_err": max(errors)}
