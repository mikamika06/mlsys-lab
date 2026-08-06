import numpy as np


def unpad(hidden_states, attention_mask):
    mask = attention_mask.astype(bool)
    return hidden_states[mask]


def pad(unpadded, attention_mask):
    mask = attention_mask.astype(bool)
    shape = (attention_mask.shape[0], attention_mask.shape[1], unpadded.shape[1])
    out = np.zeros(shape, dtype=unpadded.dtype)
    out[mask] = unpadded
    return out
