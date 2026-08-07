def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    from engine.profile import profile_layers
    from harness.ref import generate_reference_model, run_fp16_inference, run_int8_inference, get_calibration_set

    m = {"layers_profiled": 0.0, "mse_computed": 0.0}
    model = generate_reference_model(12)
    calib = get_calibration_set()
    inputs = calib[0]
    fp16_outs = run_fp16_inference(model, inputs)
    int8_outs = run_int8_inference(model, inputs, sensitive_indices=[])

    try:
        profile_res = profile_layers(model, fp16_outs, int8_outs)
    except Exception:
        return m

    if isinstance(profile_res, dict) and len(profile_res) >= 12:
        m["layers_profiled"] = float(len(profile_res))
        if all(isinstance(v, float) for v in profile_res.values()):
            m["mse_computed"] = 1.0
    return m
