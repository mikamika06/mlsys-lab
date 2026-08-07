import numpy as np

CONFIGS = [
    {"max_batch": 2, "max_len": 128, "num_heads": 4, "head_dim": 64, "num_layers": 2},
    {"max_batch": 4, "max_len": 256, "num_heads": 8, "head_dim": 128, "num_layers": 4},
]

def ref_compute_quant_error(tensor, nbits):
    qmax = (1 << nbits) - 1
    t_min = np.min(tensor)
    t_max = np.max(tensor)
    scale = (t_max - t_min) / qmax if t_max > t_min else 1.0
    quantized = np.clip(np.round((tensor - t_min) / (scale + 1e-8)), 0, qmax)
    dequantized = quantized * scale + t_min
    return float(np.mean((tensor - dequantized) ** 2))

def ref_static_cache_shapes(config):
    bs = config["max_batch"]
    ml = config["max_len"]
    nh = config["num_heads"]
    hd = config["head_dim"]
    nl = config["num_layers"]
    return [(bs, nh, ml, hd) for _ in range(nl * 2)]
