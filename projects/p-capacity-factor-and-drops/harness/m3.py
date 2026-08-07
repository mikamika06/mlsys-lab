def check(workdir):
    import sys
    import math
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from moe_drops.router import route_lossless
    import ref

    m = {"lossless_cost_ok": 0.0}
    indices = ref.get_indices(1337, 512, 4)

    mc, pad = route_lossless(indices, 4)
    mc_ref, pad_ref = ref.ref_route_lossless(indices, 4)

    if mc == mc_ref and math.isclose(pad, pad_ref, abs_tol=1e-5):
        m["lossless_cost_ok"] = 1.0

    return m
