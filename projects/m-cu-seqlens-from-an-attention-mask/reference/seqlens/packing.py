import numpy as np
from seqlens.core import extract_cu_seqlens


def unpad(padded, mask):
    cu_seqlens, lengths = extract_cu_seqlens(mask)
    unpadded_list = []
    for i, l in enumerate(lengths):
        unpadded_list.append(padded[i, :l])
    if unpadded_list:
        unpadded = np.concatenate(unpadded_list, axis=0)
    else:
        unpadded = np.zeros((0, padded.shape[-1]), dtype=padded.dtype)
    return unpadded, cu_seqlens


def pad(unpadded, cu_seqlens, original_shape):
    batch_size = len(cu_seqlens) - 1
    padded = np.zeros(original_shape, dtype=unpadded.dtype)
    for i in range(batch_size):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        length = end - start
        if length > 0:
            padded[i, :length] = unpadded[start:end]
    return padded
