def contrast_isa(native_rep, manual_rep):
    diffs = {}
    all_keys = set(native_rep.keys()).union(set(manual_rep.keys()))
    for k in all_keys:
        nv = native_rep.get(k, False)
        mv = manual_rep.get(k, False)
        if nv != mv:
            diffs[k] = {"native": nv, "manual": mv}
    return diffs
