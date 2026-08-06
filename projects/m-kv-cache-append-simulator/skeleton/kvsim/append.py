import numpy as np


class KVCacheAppendSimulator:

    def __init__(self, num_sequences, max_seq_len, block_size, num_heads, head_dim, dtype_bytes=2):
        raise NotImplementedError

    def append_tokens(self, seq_indices, num_tokens_per_seq):
        raise NotImplementedError

    def get_cache_seqlens(self):
        raise NotImplementedError

    def get_allocated_blocks(self):
        raise NotImplementedError

    def get_token_positions(self, seq_idx):
        raise NotImplementedError
