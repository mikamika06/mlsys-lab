import numpy as np

def classify_extension(inv_freq: np.ndarray) -> str:
    inv_freq = np.asarray(inv_freq, dtype=np.float64)
    n = len(inv_freq)
    if n < 2:
        return "None"
    idx = np.arange(n)
    std = 1 / (10000 ** (2 * idx / n))
    # PI: constant scaling factor
    k_vals = inv_freq / std
    k_mean = np.mean(k_vals)
    if abs(k_mean - 1.0) > 1e-6:
        return "PI"
    # NTK: geometric ratio sqrt of standard
    ratios = inv_freq[1:] / inv_freq[:-1]
    r_std_expected = std[1] / std[0]
    r_ntk_expected = np.sqrt(r_std_expected)
    if np.allclose(ratios, r_ntk_expected, atol=1e-6 * np.abs(r_ntk_expected)):
        return "NTK"
    # YaRN: linear progression in value
    diffs = np.diff(inv_freq)
    if np.std(diffs) < 1e-6 * np.mean(np.abs(diffs)):
        return "YaRN"
    return "None"
