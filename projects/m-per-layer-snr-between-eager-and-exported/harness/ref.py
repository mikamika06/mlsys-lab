import numpy as np

def make_model(num_layers=10, seed=42):
    np.random.seed(seed)
    weights = [np.random.randn(16, 16) for _ in range(num_layers)]
    return weights

def run_eager(weights, x):
    outs = []
    curr = x
    for w in weights:
        curr = np.tanh(np.dot(curr, w))
        outs.append(curr)
    return outs

def run_exported(weights, x, diverge_layer=5):
    outs = []
    curr = x
    for i, w in enumerate(weights):
        if i >= diverge_layer:
            curr = np.tanh(np.dot(curr, w)) + 0.5 * np.ones_like(curr)
        else:
            curr = np.tanh(np.dot(curr, w))
        outs.append(curr)
    return outs

def compute_snr(ref_out, exp_out):
    snrs = []
    for r, e in zip(ref_out, exp_out):
        signal = np.var(r)
        noise = np.var(r - e)
        if noise < 1e-12:
            snr = 100.0
        else:
            snr = 10.0 * np.log10(signal / noise)
        snrs.append(snr)
    return snrs

def find_diverging_layer(snrs, threshold=20.0):
    for i, s in enumerate(snrs):
        if s < threshold:
            return i
    return -1

TEST_CASES = [
    {"num_layers": 8, "diverge_layer": 3, "threshold": 15.0},
    {"num_layers": 12, "diverge_layer": 7, "threshold": 20.0},
    {"num_layers": 6, "diverge_layer": -1, "threshold": 25.0},
]
