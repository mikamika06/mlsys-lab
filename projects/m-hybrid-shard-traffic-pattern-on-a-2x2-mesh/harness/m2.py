import ref


def check(workdir):
    from meshshard.policy import diagnose_policy

    out = {"policy_matched": 0.0}
    ok = 0
    for tree, min_s in ref.POLICIES:
        want = ref.diagnose_policy(tree, min_s)
        got = diagnose_policy(tree, min_s)
        if got == want:
            ok += 1
    out["policy_matched"] = float(ok)
    return out
