import numpy as np

def compute_layer_snrs(eager_outs, exported_outs):
    snrs = []
    for r, e in zip(eager_outs, exported_outs):
        signal = np.var(r)
        noise = np.var(r - e)
        if noise < 1e-12:
            snrs.append(100.0)
        else:
            snrs.append(float(10.0 * np.log10(signal / (noise + 1e-14))))
    return snrs

def bisect_divergence(snrs, threshold=20.0):
    for i, s in enumerate(snrs):
        if s < threshold:
            return i
    return -1
