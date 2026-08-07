import ref

def check(workdir):
    from dnnlog import classify_impl, reconcile_timing
    rows = ref.get_reference_rows()
    out = {"classification_match": 0.0, "timing_match": 0.0}

    classes = [classify_impl(r["impl"]) for r in rows]
    if len(classes) == len(rows) and all(isinstance(c, str) for c in classes):
        out["classification_match"] = 1.0

    res = reconcile_timing(rows, 5.0)
    expected_sum = sum(r["time_ms"] for r in rows)
    if abs(res.get("total_kernel_time_ms", 0.0) - expected_sum) < 1e-5:
        out["timing_match"] = 1.0

    return out
