import ref


def check(workdir):
    from quantfix.hessian import validate_hessian
    out = {"hessian_checks_matched": 0.0}
    ok = 0
    for i, case in enumerate(ref.TEST_HESSIANS):
        got = validate_hessian(case["matrix"])
        want = case["valid"]
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    out["hessian_checks_matched"] = float(ok)
    return out
