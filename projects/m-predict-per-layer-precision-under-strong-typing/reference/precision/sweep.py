import numpy as np
from precision.formats import get_format_props

def run_precision_sweep(weights, formats):
    results = {}
    for fmt in formats:
        props = get_format_props(fmt)
        m_bits = props["mantissa_bits"]
        scale = 2.0 ** m_bits
        quantized = np.round(weights * scale) / scale
        mse = float(np.mean((weights - quantized) ** 2))
        max_err = float(np.max(np.abs(weights - quantized)))
        results[fmt] = {"mse": mse, "max_err": max_err}
    return results
