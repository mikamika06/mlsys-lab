def grade(sol, fx) -> dict:
    # Reference specification derived from the bias formula.
    specs = {
        'fp16':  (5, 10),
        'bf16':  (8, 7),
        'E4M3':  (4, 3),
        'E5M2':  (5, 2)
    }
    ref = {}
    for name, (exp_bits, mantissa_bits) in specs.items():
        bias = 2 ** (exp_bits - 1) - 1
        ref[name] = (exp_bits, mantissa_bits, bias)

    try:
        got = sol.identify_fp_formats()
    except Exception:
        return {"exact_match": 0.0}

    ok = 1.0 if got == ref else 0.0
    return {"exact_match": ok}
