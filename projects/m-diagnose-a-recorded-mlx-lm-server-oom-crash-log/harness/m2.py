import numpy as np
import ref


def check(workdir):
    out = {"configs_reconstructed": 0.0, "drift_evaluated": 0.0}
    try:
        from mlxdiag.quant import extract_quant_config
        from mlxdiag.drift import evaluate_weight_drift
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    cfg_ok = True
    for cfg in ref.CONFIG_SAMPLES:
        want = ref.extract_quant_config(cfg)
        got = extract_quant_config(cfg)
        if got != want:
            cfg_ok = False
            out["_note"] = f"Config mismatch. Want {want}, got {got}"
            break
    if cfg_ok:
        out["configs_reconstructed"] = 1.0

    np.random.seed(123)
    weights = np.random.randn(64, 64).astype(np.float32)
    want_drift = ref.evaluate_weight_drift(weights, bits=4, group_size=32, max_allowed_mse=0.02)
    got_drift = evaluate_weight_drift(weights, bits=4, group_size=32, max_allowed_mse=0.02)

    if (
        abs(got_drift.get("mse", 0.0) - want_drift["mse"]) < 1e-4
        and got_drift.get("exceeds_threshold") == want_drift["exceeds_threshold"]
    ):
        out["drift_evaluated"] = 1.0
    else:
        out["_note"] = f"Drift mismatch. Want {want_drift}, got {got_drift}"

    return out
