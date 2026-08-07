def check(workdir):
    import sys
    import math
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from moe_drops.model import expected_drop_rate
    import ref

    m = {"model_fit_ok": 0.0}
    val = expected_drop_rate(1.1, 8, 2048)
    val_ref = ref.ref_expected_drop_rate(1.1, 8, 2048)

    if math.isclose(val, val_ref, abs_tol=1e-5):
        m["model_fit_ok"] = 1.0

    return m
