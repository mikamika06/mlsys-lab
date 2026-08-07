import numpy as np

class KVCacheTracker:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        raise NotImplementedError

    def record(self, batch_size: int, seq_len: int) -> int:
        raise NotImplementedError

    def total_bytes(self) -> int:
        raise NotImplementedError
