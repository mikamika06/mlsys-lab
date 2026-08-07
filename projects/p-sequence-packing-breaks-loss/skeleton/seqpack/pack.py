import numpy as np


def pack_sequences(sequences, max_seq_len, pad_token_id=0):
    """Packs multiple sequence dicts into fixed-length rows."""
    raise NotImplementedError


def measure_attention_leakage(attn_weights, seq_ids):
    """Measures attention weight leaking across distinct sequence IDs."""
    raise NotImplementedError
