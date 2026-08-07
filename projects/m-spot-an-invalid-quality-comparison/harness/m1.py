import ref


def check(workdir):
    from eval.compare import check_comparison_validity, is_statistically_significant

    out = {"comparisons_checked": 0.0}
    ok = 0
    for run_a, run_b in ref.COMPARISON_PAIRS:
        want = ref.check_comparison_validity(run_a, run_b)
        got = check_comparison_validity(run_a, run_b)
        if got.get("valid") == want["valid"] and set(got.get("reasons", [])) == set(want["reasons"]):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, expected {want}"

    sig_got = is_statistically_significant(0.80, 0.01, 0.70, 0.01)
    sig_want = ref.is_statistically_significant(0.80, 0.01, 0.70, 0.01)
    if sig_got == sig_want:
        ok += 0
    else:
        out["_note"] = "is_statistically_significant mismatch"
        return out

    out["comparisons_checked"] = float(ok)
    return out
