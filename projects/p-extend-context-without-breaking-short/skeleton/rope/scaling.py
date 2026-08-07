import numpy as np

class RoPEScaling:
    def __init__(self, dim, max_len=4096, scale_type="linear", factor=1.0):
        raise NotImplementedError

    def compute_frequencies(self, seq_len):
        raise NotImplementedError

    def apply_rope(self, x, seq_len):
        raise NotImplementedError
