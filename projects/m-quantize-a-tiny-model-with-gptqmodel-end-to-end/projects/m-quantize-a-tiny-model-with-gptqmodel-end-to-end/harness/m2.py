import ref


def check(workdir):
    from quant.model import create_tiny_model
    from quant.engine import get_calibration_data, build_calibration_dataset, quantize_weights
    from quant.evaluate import compute_size_ratio, evaluate_error

    out = {"size_ratio": 0.0, "error_bounded": 0.0}
    try:
        model = create_tiny_model()
        inputs = get_calibration_data()
        calib = build_calibration_dataset(model, inputs)
        artifact = quantize_weights(model, calib, bits=4)

        ratio = compute_size_ratio(model, artifact)
        ref.compute_size_ratio(ref.create_tiny_model(), ref.quantize_weights(ref.create_tiny_model(), ref.build_calibration_dataset(ref.create_tiny_model(), ref.get_calibration_data()), bits=4))

        if ratio >= 3.5:
            out["size_ratio"] = 1.0

        if evaluate_error(model, artifact):
            out["error_bounded"] = 1.0
    except Exception as e:
        out["_note"] = f"Error in m2: {type(e).__name__}: {str(e)[:100]}"
    return out
