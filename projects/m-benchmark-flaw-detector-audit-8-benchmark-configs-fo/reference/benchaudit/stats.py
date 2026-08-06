import math
import numpy as np


def required_sample_size(ttft_samples, target_ci_pct: float = 0.02, confidence: float = 0.95) -> int:
    """Calculate required request sample size for target CI on p99 TTFT."""
    arr = np.array(ttft_samples, dtype=np.float64)
    if len(arr) == 0:
        return 0
    p = 0.99
    q99 = float(np.percentile(arr, 99))
    q98_5 = float(np.percentile(arr, 98.5))
    q99_5 = float(np.percentile(arr, 99.5))

    dq = max(q99_5 - q98_5, 1e-9)
    f_q = 0.01 / dq

    target_hw = target_ci_pct * q99
    z = 1.959963984540054

    se_num = math.sqrt(p * (1.0 - p))
    required_n = ((z * se_num) / (target_hw * f_q)) ** 2
    return int(math.ceil(required_n))
