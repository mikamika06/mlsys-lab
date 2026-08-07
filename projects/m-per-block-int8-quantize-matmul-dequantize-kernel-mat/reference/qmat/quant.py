import numpy as np

def simulate_e4m3(x: np.ndarray) -> np.ndarray:
    s = np.sign(x)
    v = np.abs(x)
    v = np.clip(v, 2**-6, 448.0)
    e = np.floor(np.log2(v))
    m = np.round((v / (2**e) - 1.0) * 8.0) / 8.0
    return s * (2**e) * (1.0 + m)

def e4m3_max_rel_error(x: np.ndarray) -> float:
    xq = simulate_e4m3(x)
    errs = np.abs(x - xq) / (np.abs(x) + 1e-9)
    return float(np.max(errs))
