def check(workdir):
    import sys
    import math
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from moe_drops.model import recommend_capacity_factor
    import ref

    m = {"recommendation_ok": 0.0}
    val = recommend_capacity_factor(16, 4096, 0.01)
    val_ref = ref.ref_recommend_capacity_factor(16, 4096, 0.01)

    if math.isclose(val, val_ref, abs_tol=1e-3):
        m["recommendation_ok"] = 1.0

    return m
