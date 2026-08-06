import numpy as np


def gather_batch_kv(kv_cache, cache_batch_idx, seq_lens):
    """Gather sequence key and value history from cache buffer using physical batch indices."""
    raise NotImplementedError
