def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from engine.profile import identify_sensitive_layers
    from harness.ref import generate_reference_model, run_fp16_inference, run_int8_inference, get_calibration_set, compute_layer_mse

    m = {"sensitive_identified": 0.0, "ranking_valid": 0.0}
    model = generate_reference_model(12)
    calib = get_calibration_set()
    fp16_outs = run_fp16_inference(model, calib[0])
    int8_outs = run_int8_inference(model, calib[0], sensitive_indices=[])
    mses = compute_layer_mse(fp16_outs, int8_outs)

    try:
        sensitive = identify_sensitive_layers(mses, top_k=3)
    except Exception:
        return m

    if isinstance(sensitive, list) and len(sensitive) == 3:
        m["sensitive_identified"] = 1.0
        if sorted(sensitive) == sorted(range(12), key=lambda i: mses[i], reverse=True)[:3]:
            m["ranking_valid"] = 1.0
    return m
