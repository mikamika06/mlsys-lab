import numpy as np

def quantize_weight(w, bits):
    if bits >= 32:
        return w.astype(np.float64, copy=True)
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    if w_max == w_min:
        return w.astype(np.float64, copy=True)
    q_max = (1 << int(bits)) - 1
    scale = (w_max - w_min) / float(q_max)
    q = np.clip(np.round((w - w_min) / scale), 0, q_max)
    return (q * scale + w_min).astype(np.float64)

def run_forward(weights, bit_recipe, inputs):
    x = inputs.astype(np.float64, copy=True)
    num_layers = len(weights)
    for i, (w, b) in enumerate(zip(weights, bit_recipe)):
        wq = quantize_weight(w, b)
        x = wq @ x
        if i < num_layers - 1:
            x = np.maximum(0.0, x)
    return x

def evaluate_model(weights, bit_recipe, inputs, ref_output=None):
    if ref_output is None:
        ref_output = run_forward(weights, [32] * len(weights), inputs)
    out = run_forward(weights, bit_recipe, inputs)
    return float(np.mean((out - ref_output) ** 2))
