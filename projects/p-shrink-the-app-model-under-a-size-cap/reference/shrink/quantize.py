import numpy as np

def palettize_tensor(tensor, clusters=16):
    flat = tensor.flatten()
    mn, mx = np.min(flat), np.max(flat)
    bins = np.linspace(mn, mx, clusters)
    idx = np.abs(flat[:, None] - bins[None, :]).argmin(axis=1)
    return bins[idx].reshape(tensor.shape).astype(tensor.dtype)

def quantize_tensor(tensor, bits=8):
    mn, mx = np.min(tensor), np.max(tensor)
    if mn == mx:
        return tensor.copy()
    scale = (mx - mn) / (2**bits - 1)
    q = np.round((tensor - mn) / scale)
    return (q * scale + mn).astype(tensor.dtype)

def prune_tensor(tensor, sparsity=0.5):
    flat = np.abs(tensor.flatten())
    thresh = np.percentile(flat, sparsity * 100)
    mask = np.abs(tensor) >= thresh
    return tensor * mask
