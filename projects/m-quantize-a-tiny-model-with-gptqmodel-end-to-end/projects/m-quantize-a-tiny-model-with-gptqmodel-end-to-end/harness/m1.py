import ref


def check(workdir):
    from quant.model import create_tiny_model
    from quant.engine import get_calibration_data, build_calibration_dataset

    out = {"calib_matched": 0.0}
    try:
        model = create_tiny_model()
        inputs = get_calibration_data()
        ref_ds = ref.build_calibration_dataset(ref.create_tiny_model(), ref.get_calibration_data())
        got_ds = build_calibration_dataset(model, inputs)
        if isinstance(got_ds, dict) and "inputs" in got_ds and "activations" in got_ds:
            if len(got_ds["inputs"]) == len(ref_ds["inputs"]):
                out["calib_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Error in m1: {type(e).__name__}: {str(e)[:100]}"
    return out
