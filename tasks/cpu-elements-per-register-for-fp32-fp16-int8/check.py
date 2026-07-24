def grade(sol, fx) -> dict:
    cases = [128, 256, 512]
    ok = 1.0
    for reg_bits in cases:
        try:
            got = sol.lanes_per_register(reg_bits)
        except Exception:
            ok = 0.0
            break
        # compute reference
        ref = {
            "float32": reg_bits // 32,
            "float16": reg_bits // 16,
            "int8":   reg_bits // 8,
        }
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
