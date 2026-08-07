import ref


def check(workdir):
    from workset.model import compute_working_set

    out = {"sizes_matched": 0.0, "configs": float(len(ref.CASES))}
    ok = 0
    for i, case in enumerate(ref.CASES):
        want = ref.compute_working_set(case["trace"], case["target"])
        got = compute_working_set(case["trace"], case["target"])
        if abs(got - want) <= 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"
    out["sizes_matched"] = float(ok)
    return out
