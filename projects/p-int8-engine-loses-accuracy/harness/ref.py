import numpy as np

def generate_reference_model(num_layers=12, seed=42):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(num_layers):
        w = rng.normal(0.0, 0.1, size=(64, 64)).astype(np.float32)
        layers.append({"id": f"layer_{i}", "weights": w, "type": "linear"})
    return layers

def run_fp16_inference(model, inputs):
    outputs = []
    curr = inputs
    for layer in model:
        curr = np.maximum(0.0, np.dot(curr, layer["weights"]))
        outputs.append(curr.copy())
    return outputs

def run_int8_inference(model, inputs, sensitive_indices=None):
    if sensitive_indices is None:
        sensitive_indices = []
    outputs = []
    curr = inputs
    for i, layer in enumerate(model):
        if i in sensitive_indices:
            val = np.maximum(0.0, np.dot(curr, layer["weights"]))
        else:
            w_q = np.round(layer["weights"] * 127.0).astype(np.int8).astype(np.float32) / 127.0
            x_q = np.round(curr * 127.0).astype(np.int8).astype(np.float32) / 127.0
            val = np.maximum(0.0, np.dot(x_q, w_q))
        outputs.append(val.copy())
        curr = val
    return outputs

def compute_layer_mse(fp16_outs, int8_outs):
    mses = []
    for f_out, i_out in zip(fp16_outs, int8_outs):
        mse = float(np.mean((f_out - i_out) ** 2))
        mses.append(mse)
    return mses

def get_calibration_set(seed=42):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 1.0, size=(16, 64)).astype(np.float32) for _ in range(5)]
