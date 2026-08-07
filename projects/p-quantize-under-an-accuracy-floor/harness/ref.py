import numpy as np


class ToyModel:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.layers = [
            rng.standard_normal((32, 32)).astype(np.float32),
            rng.standard_normal((32, 32)).astype(np.float32),
            rng.standard_normal((32, 16)).astype(np.float32),
        ]

    def forward(self, x):
        out = x
        for w in self.layers:
            out = np.tanh(np.dot(out, w))
        return out

    def size_bytes(self):
        return sum(w.nbytes for w in self.layers)


def generate_dataset(num_samples=100, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, 32)).astype(np.float32)
    y = (rng.standard_normal((num_samples, 16)) > 0).astype(np.float32)
    return x, y


def run_eval(model, x, y):
    preds = model.forward(x)
    mse = float(np.mean((preds - y) ** 2))
    accuracy = float(np.mean(np.abs(preds - y) < 0.5))
    return {"mse": mse, "accuracy": accuracy}


def get_calibration_data(x, num_samples=32):
    return x[:num_samples]


def quantize_uniform(weights, bits=8):
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))
    scale = max(abs(w_min), abs(w_max)) / qmax if qmax != 0 else 1.0
    scale = max(scale, 1e-8)
    q_weights = np.clip(np.round(weights / scale), qmin, qmax).astype(np.int8)
    return {"weights": q_weights, "scale": scale, "bits": bits}


def assign_mixed_precision(model, sensitivity_threshold=0.5):
    assignments = []
    for i, w in enumerate(model.layers):
        sens = float(np.std(w))
        bits = 4 if sens < sensitivity_threshold else 8
        assignments.append((i, bits))
    return assignments


def check_target(orig_size, quant_size, orig_acc, quant_acc, max_drop=0.01):
    size_ratio = quant_size / orig_size
    acc_drop = orig_acc - quant_acc
    return bool(size_ratio <= 0.55 and acc_drop <= max_drop)


def generate_report(results):
    return f"Report: size_ratio={results.get('size_ratio', 0):.2f}, acc_drop={results.get('acc_drop', 0):.4f}"
