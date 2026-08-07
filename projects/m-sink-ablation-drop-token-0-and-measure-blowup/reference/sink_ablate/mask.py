import numpy as np

def reconstruct_mask(num_layers, num_heads, seq_len, dump):
    """Reconstruct a boolean mask from the cache dump."""
    mask = np.zeros((num_layers, num_heads, seq_len), dtype=bool)
    for entry in dump:
        mask[entry["layer"], entry["head"], entry["kept_positions"]] = True
    return mask
