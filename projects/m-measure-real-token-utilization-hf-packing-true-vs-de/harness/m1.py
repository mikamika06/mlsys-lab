import ref


def check(workdir):
    from packutil.analyzer import measure_utilization

    datasets = ref.get_test_datasets()
    max_len = 512
    max_rel_err = 0.0
    for ds in datasets:
        want = ref.compute_oracle_utilization(ds, max_len)
        try:
            got = measure_utilization(ds, max_len)
        except Exception as e:
            return {"utilization_rel_err": 1.0, "_note": f"raised {type(e).__name__}: {e}"}
        if not isinstance(got, dict) or "packed_utilization" not in got:
            return {"utilization_rel_err": 1.0, "_note": "return value invalid format"}
        err = abs(got["packed_utilization"] - want["packed_utilization"]) / (want["packed_utilization"] + 1e-8)
        if err > max_rel_err:
            max_rel_err = err
    out = {"utilization_rel_err": float(max_rel_err)}
    if max_rel_err > 0.01:
        out["_note"] = f"max relative error {max_rel_err:.4f}"
    return out
