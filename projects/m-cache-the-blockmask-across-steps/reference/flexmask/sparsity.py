import numpy as np

def analyze_sparsity(mask_tensor, block_size=16):
    """Analyzes element and block sparsity."""
    arr = mask_tensor.detach().cpu().numpy() if hasattr(mask_tensor, "detach") else np.array(mask_tensor)
    elem_sparsity = float(np.sum(arr == 0)) / arr.size
    h, w = arr.shape[-2], arr.shape[-1]
    bh = (h + block_size - 1) // block_size
    bw = (w + block_size - 1) // block_size
    block_arr = np.zeros((bh, bw), dtype=bool)
    for i in range(bh):
        for j in range(bw):
            sub = arr[..., i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if np.all(sub == 0):
                block_arr[i, j] = True
    block_sparsity = float(np.sum(block_arr)) / block_arr.size
    return {"element_sparsity": elem_sparsity, "block_sparsity": block_sparsity}
