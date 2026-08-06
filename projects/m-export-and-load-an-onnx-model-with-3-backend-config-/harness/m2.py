import ref

def check(workdir):
    from onnxconfig.variants import make_variants
    from onnxconfig.validator import validate
    out = {"variants_matched": 0.0, "params_valid": 0.0}
    matched = 0
    valid_count = 0
    for spec in ref.MODELS:
        want = ref.generate_variants(spec)
        got = make_variants(spec)
        if got == want:
            matched += 1
        if all(validate(v) for v in got):
            valid_count += 1
    if matched == len(ref.MODELS):
        out["variants_matched"] = 1.0
    if valid_count == len(ref.MODELS):
        out["params_valid"] = 1.0
    return out
