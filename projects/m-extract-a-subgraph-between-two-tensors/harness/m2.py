import ref


def check(workdir):
    from surgery.fuse import fuse_gelu

    base = ref.make_test_graph()
    want = ref.fuse_gelu(base)
    got = fuse_gelu(base)
    out = {"fusions_matched": 0.0}
    if got == want:
        out["fusions_matched"] = 3.0
    else:
        out["_note"] = "Gelu fusion result does not match expected graph"
    return out
