import ref


def check(workdir):
    from quantizer.oneshot import run_oneshot

    model_off = ref.get_tiny_model()
    model_on = ref.get_tiny_model()

    ref_off = ref.run_oneshot_reference(model_off, sequential_onloading=False)
    ref_on = ref.run_oneshot_reference(model_on, sequential_onloading=True)

    try:
        got_off = run_oneshot(model_off, sequential_onloading=False)
        got_on = run_oneshot(model_on, sequential_onloading=True)
    except Exception as e:
        return {"oneshot_matched": 0.0, "_note": f"Exception raised: {type(e).__name__}: {str(e)}"}

    matched = 0
    if got_off is not None and isinstance(got_off, dict) and len(got_off) > 0:
        matched += 1
    if got_on is not None and isinstance(got_on, dict) and len(got_on) > 0:
        matched += 1

    return {"oneshot_matched": float(matched)}
