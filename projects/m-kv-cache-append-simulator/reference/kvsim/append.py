import numpy as np


class KVCacheAppendSimulator:

    def __init__(self, num_sequences, max_seq_len, block_size, num_heads, head_dim, dtype_bytes=2):
        self.num_sequences = num_sequences
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
        self.seqlens = np.zeros(num_sequences, dtype=np.int32)
        self.allocated_blocks = np.zeros(num_sequences, dtype=np.int32)

    def append_tokens(self, seq_indices, num_tokens_per_seq):
        seq_indices = np.asarray(seq_indices, dtype=np.int32)
        num_tokens = np.asarray(num_tokens_per_seq, dtype=np.int32)

        for seq_idx, n_tok in zip(seq_indices, num_tokens):
            if n_tok <= 0:
                continue
            cur_len = self.seqlens[seq_idx]
            new_len = cur_len + n_tok
            if new_len > self.max_seq_len:
                raise ValueError(f"Sequence {seq_idx} length {new_len} exceeds max {self.max_seq_len}")
            self.seqlens[seq_idx] = new_len
            needed_blocks = (new_len + self.block_size - 1) // self.block_size
            self.allocated_blocks[seq_idx] = needed_blocks

    def get_cache_seqlens(self):
        return self.seqlens.copy()

    def get_allocated_blocks(self):
        return self.allocated_blocks.copy()

    def get_token_positions(self, seq_idx):
        length = int(self.seqlens[seq_idx])
        if length == 0:
            return np.empty((0, 2), dtype=np.int32)
        token_indices = np.arange(length, dtype=np.int32)
        block_indices = token_indices // self.block_size
        block_offsets = token_indices % self.block_size
        return np.column_stack((block_indices, block_offsets))
