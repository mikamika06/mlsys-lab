import ref


def check(workdir):
    from torchprof.profile import measure_reduction
    model = ref.SimpleModel()
    inputs = ref.generate_inputs()

    try:
        res = measure_reduction(model, inputs)
    except Exception as e:
        return {"metrics_matched": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    if not isinstance(res, dict):
        return {"metrics_matched": 0.0, "_note": "did not return a dict"}

    ref_m = ref.get_reference_metrics()
    diff = abs(res.get("correctness", 999.0) - ref_m["correctness"])
    if diff < 1e-4 and "op_reduction_ratio" in res and "time_reduction_ratio" in res:
        return {"metrics_matched": 1.0}
    return {"metrics_matched": 0.0, "_note": f"metrics did not match expected values, got {res}"}
