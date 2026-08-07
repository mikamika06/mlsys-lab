import ref
import numpy as np


def check(workdir):
    from mxquant.quant import quantize_per_group

    w = ref.generate_test_data()
    group_size = 32
    bits = 4

    try:
        got_q, got_s, got_b = quantize_per_group(w, group_size, bits)
        ref_q, ref_s, ref_b = ref.quantize_per_group_ref(w, group_size, bits) if hasattr(ref, "quantize_per_group_ref") else _fallback_ref(w, group_size, bits)
    except Exception as e:
        return {"formulas_matched": 0.0, "_note": f"Exception raised: {e}"}

    match_q = np.allclose(got_q, ref_q, atol=1e-5)
    match_s = np.allclose(got_s, ref_s, atol=1e-5)
    match_b = np.allclose(got_b, ref_b, atol=1e-5)

    if match_q and match_s and match_b:
        return {"formulas_matched": 1.0}
    return {"formulas_matched": 0.0, "_note": "Outputs did not match reference formula"}


def _fallback_ref(weights, group_size, bits):
    in_features = weights.shape[1]
    num_groups = in_features // group_size
    reshaped = weights.reshape(weights.shape[0], num_groups, group_size)
    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))
    w_min = reshaped.min(axis=-1, keepdims=True)
    w_max = reshaped.max(axis=-1, keepdims=True)
    scale = (w_max - w_min) / (max_val - min_val)
    scale = np.where(scale == 0, 1e-5, scale)
    bias = w_min
    quantized = np.clip(np.round((reshaped - bias) / scale), min_val, max_val).astype(np.int32)
    return quantized, scale, bias
