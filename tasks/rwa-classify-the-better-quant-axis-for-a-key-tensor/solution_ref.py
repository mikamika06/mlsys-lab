import numpy as np

def classify_quant_axis(K: np.ndarray) -> int:
    # Compute per‑row and per‑column dynamic ranges
    row_ranges = np.ptp(K, axis=1)
    col_ranges = np.ptp(K, axis=0)
    var_rows = np.var(row_ranges)
    var_cols = np.var(col_ranges)
    return 0 if var_rows > var_cols else 1
