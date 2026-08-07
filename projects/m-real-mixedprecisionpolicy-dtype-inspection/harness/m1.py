import ref

def check(workdir):
    from mpinspect.policy import inspect_policy
    ok = 0
    out = {"policy_matched": 0.0, "configs": float(len(ref.POLICIES))}
    for i, p in enumerate(ref.POLICIES):
        want = ref.inspect_policy(p)
        got = inspect_policy(p)
        if got == want:
            ok += 1
    out["policy_matched"] = float(ok)
    return out
