import numpy as np

def compute_quant_error(tensor, nbits):
    qmax = (1 << nbits) - 1
    t_min = np.min(tensor)
    t_max = np.max(tensor)
    scale = (t_max - t_min) / qmax if t_max > t_min else 1.0
    quantized = np.clip(np.round((tensor - t_min) / (scale + 1e-8)), 0, qmax)
    dequantized = quantized * scale + t_min
    return float(np.mean((tensor - dequantized) ** 2))
