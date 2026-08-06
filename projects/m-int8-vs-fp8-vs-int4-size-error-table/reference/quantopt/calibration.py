import numpy as np


def evaluate_domain_calibration(weights, in_domain_calib, out_domain_calib, eval_data):
    w_scale = float(np.max(np.abs(weights)))
    w_scale = w_scale if w_scale > 0 else 1.0
    q_w = np.clip(np.round((weights / w_scale) * 127.0), -128, 127) * (
        w_scale / 127.0
    )

    orig_bytes = weights.size * 2 + eval_data.size * 2
    quant_bytes = weights.size * 1 + eval_data.size * 1
    size_ratio = float(quant_bytes / orig_bytes)

    ref_out = eval_data @ weights.T

    results = {}
    for domain_name, calib_data in [
        ("in_domain", in_domain_calib),
        ("out_domain", out_domain_calib),
    ]:
        a_scale = float(np.max(np.abs(calib_data)))
        a_scale = a_scale if a_scale > 0 else 1.0

        q_act = np.clip(np.round((eval_data / a_scale) * 127.0), -128, 127) * (
            a_scale / 127.0
        )
        quant_out = q_act @ q_w.T

        mse = float(np.mean((ref_out - quant_out) ** 2))

        results[domain_name] = {
            "act_scale": a_scale,
            "eval_mse": mse,
            "size_ratio": size_ratio,
        }

    results["domain_gap"] = float(
        results["out_domain"]["eval_mse"] - results["in_domain"]["eval_mse"]
    )
    return results
