import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"best_alpha_matches": 0.0, "quant_mse_improved": 0.0}

    try:
        from smoothquant.autotune import sweep_alpha_per_layer
    except Exception as e:
        out["_note"] = f"Failed to import smoothquant.autotune: {e}"
        return out

    activations, weights = ref.generate_synthetic_model_data(777)
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

    want_res = ref.ref_sweep_alpha_per_layer(activations, weights, candidates)

    try:
        got_res = sweep_alpha_per_layer(activations, weights, candidates)
    except Exception as e:
        out["_note"] = f"sweep_alpha_per_layer raised exception: {e}"
        return out

    alpha_ok = True
    mse_ok = True

    for name in activations:
        if name not in got_res:
            alpha_ok = False
            break
        got_alpha = got_res[name]["alpha"]
        want_alpha = want_res[name]["alpha"]

        if not np.isclose(got_alpha, want_alpha, atol=1e-5):
            alpha_ok = False

        ref_out = activations[name] @ weights[name].T
        X_raw_q = ref.ref_quantize_int8(activations[name], axis=None)
        W_raw_q = ref.ref_quantize_int8(weights[name], axis=1)
        raw_mse = float(np.mean((ref_out - (X_raw_q @ W_raw_q.T)) ** 2))

        if got_res[name]["mse"] >= raw_mse:
            mse_ok = False

    if alpha_ok:
        out["best_alpha_matches"] = 1.0
    else:
        out["_note"] = "Autotune selected incorrect optimal alpha"

    if mse_ok:
        out["quant_mse_improved"] = 1.0
    elif "_note" not in out:
        out["_note"] = "Autotuned quantization MSE did not beat naive quantization MSE"

    return out
