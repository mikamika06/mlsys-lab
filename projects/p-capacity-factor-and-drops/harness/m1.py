def check(workdir):
    import sys
    import math
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from moe_drops.router import compute_dropped_fraction
    import ref

    m = {"drop_rate_ok": 0.0}
    indices = ref.get_indices(42, 1024, 8)

    val = compute_dropped_fraction(indices, 8, 1.25)
    val_ref = ref.ref_compute_dropped_fraction(indices, 8, 1.25)

    if math.isclose(val, val_ref, abs_tol=1e-5):
        m["drop_rate_ok"] = 1.0

    return m
