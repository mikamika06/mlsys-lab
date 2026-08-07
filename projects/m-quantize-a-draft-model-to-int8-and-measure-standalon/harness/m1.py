import ref


def check(workdir):
    from draft.quantize import quantize_weights_int8, measure_standalone_latency

    weights, inputs = ref.get_test_weights_and_inputs()
    int8_data = quantize_weights_int8(weights)

    if not isinstance(int8_data, dict) or "qweights" not in int8_data or "scale" not in int8_data:
        return {"latency_ratio": 1.0, "reconstruction_error_ok": 0.0, "_note": "quantize_weights_int8 returned invalid keys"}

    res = measure_standalone_latency(weights, int8_data, inputs, iterations=10)

    ratio = float(res.get("latency_ratio", 1.0))
    max_err = float(res.get("max_error", 999.0))

    reconstruction_ok = 1.0 if max_err < 0.1 else 0.0

    return {
        "latency_ratio": ratio,
        "reconstruction_error_ok": reconstruction_ok
    }
