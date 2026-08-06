import ref

def check(workdir):
    from stallprof.analyzer import identify_imbalanced_branch
    out = {"branch_matched": 0.0}
    ok = 0
    for k in ref.KERNELS:
        want = ref.identify_imbalanced_branch(k)
        got = identify_imbalanced_branch(k)
        if got == want:
            ok += 1
    if ok == len(ref.KERNELS):
        out["branch_matched"] = 1.0
    else:
        out["_note"] = "mismatch in branch identification"
    return out
