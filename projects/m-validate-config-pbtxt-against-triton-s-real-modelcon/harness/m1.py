import ref

def check(workdir):
    from tritonval.validator import validate_config

    out = {"configs_matched": 0.0}
    ok = 0
    total = len(ref.CONFIGS)
    for i, cdata in enumerate(ref.CONFIGS):
        got = validate_config(cdata["text"])
        want_valid = cdata["expected_valid"]
        if got.get("valid") == want_valid:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got valid={got.get('valid')}, want {want_valid}"

    out["configs_matched"] = float(ok)
    return out
