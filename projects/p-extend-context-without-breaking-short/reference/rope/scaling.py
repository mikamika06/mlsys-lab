import numpy as np

class RoPEScaling:
    def __init__(self, dim, max_len=4096, scale_type="linear", factor=4.0):
        self.dim = dim
        self.max_len = max_len
        self.scale_type = scale_type
        self.factor = factor

    def compute_frequencies(self, seq_len):
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.dim, 2) / self.dim))
        if self.scale_type == "linear":
            inv_freq = inv_freq / self.factor
        elif self.scale_type == "yarn":
            low = 1.0
            high = float(self.max_len)
            inv_freq = inv_freq / (self.factor * np.clip((self.max_len / seq_len), 1.0, self.factor))
        return inv_freq

    def apply_rope(self, x, seq_len):
        inv_freq = self.compute_frequencies(seq_len)
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([np.cos(freqs), np.sin(freqs)], axis=-1)
        return x * emb[:seq_len, :x.shape[-1]]
