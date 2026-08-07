import ref


def check(workdir):
    from roof.formula import classify_kernel

    out = {"formula_matched": 0.0}
    ok = 0
    total = len(ref.PAIRS)
    for p in ref.PAIRS:
        ridge = p["peak_flop"] / p["peak_bw"]
        want_intensity = ref.compute_intensity(p["bytes"], p["bytes"] * 0.5, p["flops"])
        want_class = ref.classify_bound(want_intensity, ridge)
        got_class = classify_kernel(p["bytes"], p["bytes"] * 0.5, p["flops"], ridge)
        if got_class == want_class:
            ok += 1
    if ok == total:
        out["formula_matched"] = 1.0
    else:
        out["_note"] = f"matched {ok} of {total} pairs correctly"
    return out
