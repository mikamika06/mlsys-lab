import ref


def check(workdir):
    from ggufconv.feasibility import check_feasibility

    out = {
        "feasibility_matched": 0.0,
        "total_cases": float(len(ref.FILES_TEST_CASES)),
    }
    ok = 0
    for i, (files, want) in enumerate(ref.FILES_TEST_CASES):
        got = check_feasibility(files)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"
    out["feasibility_matched"] = float(ok)
    return out
