import numpy as np
import ref


def check(workdir):
    from quanteval.modes import evaluate_mode_output
    from quanteval.table import calculate_model_size

    out = {"modes_quantized": 0.0, "sizes_matched": 0.0}
    layers = ref.make_sample_layers(seed=123)
    x = ref.make_test_inputs(seed=123)[0]

    modes = ["fp32", "fp16", "dynamic_int8", "full_int8"]
    ok_modes = 0
    for mode in modes:
        try:
            res = evaluate_mode_output(
                layers[0]["weights"], layers[0]["bias"], x, mode, layers[0]["calibration_range"]
            )
            if isinstance(res, np.ndarray) and res.shape == (x.shape[0], layers[0]["weights"].shape[0]):
                ok_modes += 1
        except Exception as e:
            out["_note"] = f"mode {mode} failed: {e}"

    out["modes_quantized"] = float(ok_modes)

    sizes_ok = True
    for mode in modes:
        expected_size = sum(
            ref.calculate_layer_size_ref(l["weights"], l["bias"], mode) for l in layers
        )
        got_size = calculate_model_size(layers, mode)
        if expected_size != got_size:
            sizes_ok = False
            out["_note"] = f"size mismatch for {mode}: got {got_size}, expected {expected_size}"
            break

    if sizes_ok:
        out["sizes_matched"] = 1.0

    return out
