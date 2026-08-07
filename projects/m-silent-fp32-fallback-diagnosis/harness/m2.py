import ref


def check(workdir):
    from fallbackdiag.quant import select_group_size
    out = {"group_size_matched": 0.0}
    ok = 0
    for w in ref.WEIGHTS_LIST:
        want = ref.select_group_size(w, 0.05)
        got = select_group_size(w, 0.05)
        if got == want:
            ok += 1
    out["group_size_matched"] = float(ok == len(ref.WEIGHTS_LIST))
    return out
