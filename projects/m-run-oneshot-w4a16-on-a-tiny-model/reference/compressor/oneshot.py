import numpy as np


def run_oneshot_w4a16(model, tokenizer, recipes):
    compressed = {}
    for name, w in model.items():
        w_f = w.astype(np.float32)
        mn = float(np.min(w_f))
        mx = float(np.max(w_f))
        scale = (mx - mn) / 15.0 if mx != mn else 1.0
        zp = round(-mn / scale) if scale != 0 else 0
        zp = max(0, min(15, zp))
        q = np.clip(np.round(w_f / scale + zp), 0, 15).astype(np.uint8)
        compressed[name] = {
            "scale": scale,
            "zero_point": zp,
            "quantized": q,
            "size": q.nbytes,
        }
    return compressed
