import ref


def check(workdir):
    from ctxplan.plan import build_sharing_plan
    out = {"plans_matched": 0.0}
    ok = 0
    for tensors, budget in ref.CONFIGS:
        want = ref.build_sharing_plan(tensors, budget)
        try:
            got = build_sharing_plan(tensors, budget)
            if sorted(got) == sorted(want):
                ok += 1
        except Exception:
            pass
    out["plans_matched"] = float(ok)
    return out
