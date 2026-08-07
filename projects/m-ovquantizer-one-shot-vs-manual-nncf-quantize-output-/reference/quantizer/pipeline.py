import numpy as np
from quantizer.opset import identify_intermediate_opset, validate_opset_compatibility


def _quantize_tensor(tensor, num_bits=8, symmetric=True):
    val_min = float(np.min(tensor))
    val_max = float(np.max(tensor))
    if symmetric:
        max_abs = max(abs(val_min), abs(val_max))
        scale = max_abs / ((2 ** (num_bits - 1)) - 1) if max_abs > 1e-8 else 1.0
        zero_point = 0
        q_min = -(2 ** (num_bits - 1))
        q_max = (2 ** (num_bits - 1)) - 1
    else:
        scale = (val_max - val_min) / ((2**num_bits) - 1) if (val_max - val_min) > 1e-8 else 1.0
        zero_point = round(-val_min / scale) if scale > 0 else 0
        q_min = 0
        q_max = (2**num_bits) - 1

    q_raw = np.round(tensor / scale) + zero_point
    q_clamped = np.clip(q_raw, q_min, q_max)
    dq_tensor = (q_clamped - zero_point) * scale
    return dq_tensor, scale, zero_point


def ov_quantizer_one_shot(model_weights, calibration_data, config):
    """Perform one-shot OVQuantizer execution."""
    nodes = config.get("nodes", [])
    opset_map = config.get("graph_opset_map", {})
    effective_opset = identify_intermediate_opset(nodes, opset_map)
    compat = validate_opset_compatibility(effective_opset)
    if not compat["is_valid"]:
        raise ValueError(f"Unsupported opset: {effective_opset}")

    num_bits = config.get("num_bits", 8)
    symmetric = config.get("symmetric", True)

    calib_concat = np.concatenate([np.asarray(d) for d in calibration_data], axis=0)
    w_q, w_scale, w_zp = _quantize_tensor(np.asarray(model_weights), num_bits, symmetric)
    a_q, a_scale, a_zp = _quantize_tensor(calib_concat, num_bits, symmetric)

    def forward(x):
        x_arr = np.asarray(x)
        x_q = np.clip(np.round(x_arr / a_scale) + a_zp, -128, 127) if symmetric else np.clip(np.round(x_arr / a_scale) + a_zp, 0, 255)
        x_dq = (x_q - a_zp) * a_scale
        return np.dot(x_dq, w_q.T)

    return {
        "forward": forward,
        "effective_opset": effective_opset,
        "w_scale": w_scale,
        "a_scale": a_scale,
    }


def manual_nncf_quantize(model_weights, calibration_data, config):
    """Perform manual NNCF quantization pipeline."""
    nodes = config.get("nodes", [])
    opset_map = config.get("graph_opset_map", {})
    effective_opset = identify_intermediate_opset(nodes, opset_map)
    compat = validate_opset_compatibility(effective_opset)
    if not compat["is_valid"]:
        raise ValueError(f"Unsupported opset: {effective_opset}")

    num_bits = config.get("num_bits", 8)
    symmetric = config.get("symmetric", True)

    calib_concat = np.concatenate([np.asarray(d) for d in calibration_data], axis=0)
    w_q, w_scale, w_zp = _quantize_tensor(np.asarray(model_weights), num_bits, symmetric)
    a_q, a_scale, a_zp = _quantize_tensor(calib_concat, num_bits, symmetric)

    def forward(x):
        x_arr = np.asarray(x)
        x_q = np.clip(np.round(x_arr / a_scale) + a_zp, -128, 127) if symmetric else np.clip(np.round(x_arr / a_scale) + a_zp, 0, 255)
        x_dq = (x_q - a_zp) * a_scale
        return np.dot(x_dq, w_q.T)

    return {
        "forward": forward,
        "effective_opset": effective_opset,
        "w_scale": w_scale,
        "a_scale": a_scale,
    }


def verify_output_parity(weights, inputs, config):
    """Compare outputs of one-shot vs manual quantization pipelines."""
    calib = config.get("calibration_data", [inputs])
    res_one_shot = ov_quantizer_one_shot(weights, calib, config)
    res_manual = manual_nncf_quantize(weights, calib, config)

    out_one_shot = res_one_shot["forward"](inputs)
    out_manual = res_manual["forward"](inputs)

    diff = np.abs(out_one_shot - out_manual)
    denom = np.abs(out_one_shot) + 1e-8
    rel_err = float(np.max(diff / denom))

    return {
        "rel_err": rel_err,
        "one_shot_opset": res_one_shot["effective_opset"],
        "manual_opset": res_manual["effective_opset"],
        "is_exact_match": bool(rel_err <= 1e-6),
    }
