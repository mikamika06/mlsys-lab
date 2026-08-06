import numpy as np


def get_layer_masks(tensors, sparsity):
    out = {}
    for k, v in tensors.items():
        size = v.size
        num_zeros = int(size * sparsity)
        mask = np.ones(size, dtype=bool)
        if num_zeros > 0:
            idx = np.argsort(np.abs(v.flatten()))
            mask[idx[:num_zeros]] = False
        out[k] = mask.reshape(v.shape)
    return out


def get_global_masks(tensors, sparsity):
    total_size = sum(v.size for v in tensors.values())
    num_zeros = int(total_size * sparsity)
    
    flat_arrays = []
    lengths = []
    for k, v in tensors.items():
        flat = v.flatten()
        flat_arrays.append(flat)
        lengths.append(len(flat))
        
    if not flat_arrays:
        return {}
        
    all_vals = np.concatenate(flat_arrays)
    mask = np.ones(len(all_vals), dtype=bool)
    
    if num_zeros > 0:
        idx = np.argsort(np.abs(all_vals))
        mask[idx[:num_zeros]] = False
        
    out = {}
    offset = 0
    for i, (k, v) in enumerate(tensors.items()):
        length = lengths[i]
        out[k] = mask[offset:offset+length].reshape(v.shape)
        offset += length
    return out
