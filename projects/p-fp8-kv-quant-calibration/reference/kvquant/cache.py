import numpy as np

class KVCacheTracker:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.recorded_tokens = 0
        self.bytes_per_token_fp16 = num_layers * 2 * num_heads * head_dim * 2

    def record(self, batch_size: int, seq_len: int) -> int:
        tokens = batch_size * seq_len
        self.recorded_tokens += tokens
        return tokens * self.bytes_per_token_fp16

    def total_bytes(self) -> int:
        return self.recorded_tokens * self.bytes_per_token_fp16
