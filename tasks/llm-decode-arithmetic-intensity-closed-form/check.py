def grade(sol, fx) -> dict:
    import numpy as np

    def reference(num_layers, hidden_size, num_heads, seq_len):
        H = hidden_size
        N_h = num_heads
        S = seq_len
        flops = 12 * H ** 2 + 2 * H * S
        bytes_ = 48 * H ** 2 + (8 * S * H) / N_h + 4 * H
        return flops / bytes_

    cases = [
        (1, 768, 12, 128),
        (24, 4096, 32, 256),
        (2, 512, 8, 64),
        (10, 2048, 16, 512),
        (3, 1024, 16, 100),
    ]

    rel_errs = []
    type_ok = True

    for num_layers, H, N_h, S in cases:
        try:
            got = sol.arithmetic_intensity(num_layers, H, N_h, S)
        except Exception:
            return {"rel_err": 1.0, "type_ok": False}
        if not isinstance(got, (float, np.floating)):
            type_ok = False
        ref_val = reference(num_layers, H, N_h, S)
        rel_errs.append(abs(got - ref_val) / abs(ref_val))

    max_rel_err = max(rel_errs) if rel_errs else 1.0
    return {"rel_err": max_rel_err, "type_ok": type_ok}
