def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from quant.evaluator import measure_quant_error
    import ref

    m = {"error_measured_ok": 0.0}
    weights = ref.get_sample_weights()
    err = measure_quant_error(weights, "fp4", 32)
    if isinstance(err, float) and err >= 0.0:
        m["error_measured_ok"] = 1.0
    return m
