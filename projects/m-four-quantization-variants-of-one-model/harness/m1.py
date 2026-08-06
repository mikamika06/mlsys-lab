import ref


def check(workdir):
    from edgequant.variants import build_variants
    spec = ref.get_model_spec()
    want = ref.generate_variants(spec)
    got = build_variants(spec)
    out = {"variants_matched": 0.0}
    if not isinstance(got, dict):
        out["_note"] = "build_variants did not return a dict"
        return out
    match_count = 0
    for k in ["fp32", "fp16", "dynamic", "int8_full"]:
        if k in got and k in want:
            if got[k].get("io_dtype") == want[k]["io_dtype"] and got[k].get("quantized") == want[k]["quantized"]:
                match_count += 1
    out["variants_matched"] = float(match_count)
    return out
