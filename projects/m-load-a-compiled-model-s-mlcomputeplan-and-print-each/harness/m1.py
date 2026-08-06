import ref


def check(workdir):
    from mlplan.plan import parse_compute_plan

    out = {"plans_parsed": 0.0, "total_plans": float(len(ref.PLANS))}
    matched = 0
    for i, p in enumerate(ref.PLANS):
        want = ref.parse_compute_plan(p)
        got = parse_compute_plan(p)
        if got == want:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"plan {i}: got {got[:2]}, expected {want[:2]}"
    out["plans_parsed"] = float(matched)
    return out
