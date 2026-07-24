import numpy as np

def kv_cache_quantize(keys_fp16: np.ndarray, values_fp16: np.ndarray):
    """TODO: This implementation uses a single global scale for all rows,
which is incorrect for per‑row symmetric quantization and leads to
high KL divergence."""
    raise NotImplementedError('your code here')
