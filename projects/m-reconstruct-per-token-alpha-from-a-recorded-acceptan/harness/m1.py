import ref


def check(workdir):
    from specalpha.reconstruct import reconstruct_alphas
    out = {"alpha_matched": 0.0, "configs": float(len(ref.PROFILES))}
    ok = 0
    for i, profile in enumerate(ref.PROFILES):
        want = ref.generate_reference_alphas(profile)
        got = reconstruct_alphas(profile["histogram"], profile["max_k"])
        if len(got) == len(want) and all(abs(g - w) < 1e-5 for g, w in zip(got, want)):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"profile {i}: got {got}, reference {want}"
    out["alpha_matched"] = float(ok)
    return out
