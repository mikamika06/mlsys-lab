import ref


def check(workdir):
    from flopdiag.diagnose import diagnose_row
    out = {"diagnosis_match": 0.0}
    tests = ref.get_m1_tests()
    ok = True
    for op_name, flops, reg, expected in tests:
        got = diagnose_row(op_name, flops, reg)
        if got != expected:
            ok = False
            out["_note"] = f"op {op_name}: got {got}, want {expected}"
            break
    if ok:
        out["diagnosis_match"] = 1.0
    return out
