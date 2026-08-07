import ref


def check(workdir):
    from quant.engine import determine_group_size

    out = {"groups_matched": 0.0, "cases": float(len(ref.BUDGET_CASES))}
    ok = 0
    for i, case in enumerate(ref.BUDGET_CASES):
        want = ref.compute_group_size(case)
        got = determine_group_size(case)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"
    out["groups_matched"] = float(ok)
    return out
