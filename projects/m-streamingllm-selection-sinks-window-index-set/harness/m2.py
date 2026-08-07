import ref

def check(workdir):
    from streamkv.h2o import h2o_heavy_hitters
    from streamkv.snapkv import snapkv_pool_scores

    out = {"h2o_matched": 0.0, "snapkv_matched": 0.0}

    h2o_ok = 0
    for mat in ref.MATRICES:
        want = ref.h2o_heavy_hitters(mat, 2)
        got = h2o_heavy_hitters(mat, 2)
        if sorted(got) == sorted(want):
            h2o_ok += 1
    if h2o_ok == len(ref.MATRICES):
        out["h2o_matched"] = 1.0

    snap_ok = 0
    for mat in ref.MATRICES:
        want = ref.snapkv_pool_scores(mat, 4, 2)
        got = snapkv_pool_scores(mat, 4, 2)
        if sorted(got) == sorted(want):
            snap_ok += 1
    if snap_ok == len(ref.MATRICES):
        out["snapkv_matched"] = 1.0

    return out
