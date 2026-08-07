import numpy as np

def check(workdir):
    from kvquant.quantize import quantize_fp8, dequantize_fp8
    from kvquant.calib import calibrate_scale
    m = {"perplexity_delta_ok": 0.0}

    np.random.seed(777)
    dialog = [np.random.randn(32, 64).astype(np.float32) for _ in range(20)]
    scale = calibrate_scale(dialog)

    diffs = []
    for t in dialog:
        q = quantize_fp8(t, scale)
        dq = dequantize_fp8(q, scale)
        diffs.append(np.mean((t - dq)**2))

    if np.mean(diffs) < 0.02:
        m["perplexity_delta_ok"] = 1.0

    return m
