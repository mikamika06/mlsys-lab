def check(workdir):
    import sys
    import math
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from moe_drops.quality import quality_penalty
    import ref

    m = {"quality_impact_ok": 0.0}
    val = quality_penalty(0.05)
    val_ref = ref.ref_quality_penalty(0.05)

    if math.isclose(val, val_ref, abs_tol=1e-5):
        m["quality_impact_ok"] = 1.0

    return m
