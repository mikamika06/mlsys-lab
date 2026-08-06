import ref


def check(workdir):
    from moecov.coverage import measure_coverage

    out = {"coverage_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.measure_coverage(cfg["routing"], cfg["num_experts"])
        got = measure_coverage(cfg["routing"], cfg["num_experts"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["coverage_matched"] = float(ok)
    return out
