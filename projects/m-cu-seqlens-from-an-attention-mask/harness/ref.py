import numpy as np


def generate_cases():
    np.random.seed(42)
    cases = []
    for _ in range(5):
        batch_size = np.random.randint(2, 5)
        max_len = np.random.randint(8, 16)
        mask = np.zeros((batch_size, max_len), dtype=np.int32)
        for i in range(batch_size):
            length = np.random.randint(3, max_len + 1)
            mask[i, :length] = 1
        cases.append(mask)
    return cases


def ref_compute_cu_seqlens(attention_mask):
    lengths = np.sum(attention_mask, axis=1)
    cu = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    return cu


def ref_unpad(hidden_states, attention_mask):
    mask_bool = attention_mask.astype(bool)
    unpadded = hidden_states[mask_bool]
    return unpadded


def ref_pad(unpadded, attention_mask, fill_value=0.0):
    batch, seq_len = attention_mask.shape
    hidden_dim = unpadded.shape[-1]
    padded = np.full((batch, seq_len, hidden_dim), fill_value, dtype=unpadded.dtype)
    mask_bool = attention_mask.astype(bool)
    padded[mask_bool] = unpadded
    return padded


def ref_detect_leakage(attention_weights, cu_seqlens):
    total = cu_seqlens[-1]
    if attention_weights.shape[-2] != total or attention_weights.shape[-1] != total:
        return True
    valid_mask = np.zeros((total, total), dtype=bool)
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i+1]
        valid_mask[s:e, s:e] = True
    leaked = np.any((attention_weights > 0) & (~valid_mask))
    return bool(leaked)
