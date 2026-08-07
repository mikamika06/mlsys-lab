import ref

def check(workdir):
    from scaler_lab.run_loop import run_scaling_loop
    model, opt, stream = ref.get_test_fixture()
    try:
        scales = run_scaling_loop(model, opt, stream, inject_inf_steps=[1, 3])
        if isinstance(scales, list) and len(scales) == len(stream):
            return {"inf_handled": 1.0}
    except Exception as e:
        return {"inf_handled": 0.0, "_note": f"failed: {e}"}
    return {"inf_handled": 0.0}
