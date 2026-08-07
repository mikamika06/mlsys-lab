import numpy as np
import ref

def check(workdir):
    from zero_diag.fragmentation import compute_fragmentation_curve
    from zero_diag.planner import compute_optimal_reduce_bucket_size

    out = {
        "fragmentation_matched": 0.0,
        "optimal_bucket_matched": 0.0
    }

    frag_ok = 0
    total_frag = len(ref.PARAM_SHAPE_FIXTURES)
    for shapes in ref.PARAM_SHAPE_FIXTURES:
        want = ref.compute_fragmentation_curve(shapes)
        got = compute_fragmentation_curve(shapes)

        if (got.get("total_overhead_bytes") == want["total_overhead_bytes"] and
            np.allclose(got.get("cumulative_padded_bytes", []), want["cumulative_padded_bytes"]) and
            np.allclose(got.get("fragmentation_ratio_curve", []), want["fragmentation_ratio_curve"], rtol=1e-3)):
            frag_ok += 1
        elif "_note" not in out:
            out["_note"] = f"Fragmentation mismatch. Want {want}, got {got}"

    if frag_ok == total_frag:
        out["fragmentation_matched"] = 1.0

    plan_ok = 0
    total_plan = len(ref.PLANNER_FIXTURES)
    for kwargs in ref.PLANNER_FIXTURES:
        want_b = ref.compute_optimal_reduce_bucket_size(**kwargs)
        got_b = compute_optimal_reduce_bucket_size(**kwargs)

        if abs(got_b - want_b) / float(want_b) <= 0.05:
            plan_ok += 1
        elif "_note" not in out:
            out["_note"] = f"Planner mismatch. Want {want_b}, got {got_b}"

    if plan_ok == total_plan:
        out["optimal_bucket_matched"] = 1.0

    return out
