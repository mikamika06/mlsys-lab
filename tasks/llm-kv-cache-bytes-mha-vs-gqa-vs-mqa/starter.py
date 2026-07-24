import numpy as np

def kv_cache_bytes(layers: int, ctx_len: int, d_model: int, num_heads: int, groups: int=1, dtype=np.float32) -> dict:
    """Broken implementation – uses the number of heads instead of
the number of groups for GQA, so the returned size is incorrect."""
    raise NotImplementedError('your code here')
