import ref

def check(workdir):
    from engineprof.profile import aggregate_profile
    out = {"profiles_matched": 0.0, "profiles": float(len(ref.PROFILES))}
    ok = 0
    for i, p in enumerate(ref.PROFILES):
        want = ref.aggregate_profile(p)
        got = aggregate_profile(p)
        if got == want:
            ok += 1
    out["profiles_matched"] = float(ok)
    return out
