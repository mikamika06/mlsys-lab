import ref


def check(workdir):
    from nmvalidate.validator import validate_nm_sparsity
    out = {"validations_matched": 0.0, "total": float(len(ref.TEST_CASES))}
    ok = 0
    for i, tc in enumerate(ref.TEST_CASES):
        try:
            got = validate_nm_sparsity(tc["weight"], tc["n"], tc["m"], tc["dim"])
            if got == tc["valid"]:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {i}: got {got}, expected {tc['valid']}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}"
    out["validations_matched"] = float(ok)
    return out
