import numpy as np


def run_oneshot_w4a16(model_config, weights):
    quantized_weights = {}
    for name, w in weights.items():
        w_f32 = w.astype(np.float32)
        min_val = float(np.min(w_f32))
        max_val = float(np.max(w_f32))
        scale = (max_val - min_val) / 15.0 if max_val != min_val else 1.0
        zero_point = round(-min_val / scale) if scale != 0 else 0
        zero_point = max(0, min(15, zero_point))
        q = np.clip(np.round(w_f32 / scale + zero_point), 0, 15).astype(np.uint8)
        packed = (q[..., ::2] << 4) | q[..., 1::2] if q.shape[-1] % 2 == 0 else q
        quantized_weights[name] = {
            "packed": packed,
            "scale": scale,
            "zero_point": zero_point,
            "shape": w.shape,
            "bits": 4,
        }
    return quantized_weights
