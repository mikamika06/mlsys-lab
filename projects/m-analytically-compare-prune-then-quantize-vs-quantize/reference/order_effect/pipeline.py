import numpy as np
from order_effect.analytic import quantize_tensor


def run_joint_compression(W, X, sparsity, num_bits, block_size=64):
    out_dim, in_dim = W.shape
    W_compressed = np.copy(W).astype(np.float64)
    
    XXT = X @ X.T
    damp = 1e-4 * np.trace(XXT) / in_dim
    H = XXT + damp * np.eye(in_dim)
    H_inv = np.linalg.inv(H)
    
    k = int(np.round(in_dim * sparsity))
    
    for b_start in range(0, in_dim, block_size):
        b_end = min(b_start + block_size, in_dim)
        b_len = b_end - b_start
        
        H_inv_block = H_inv[b_start:b_end, b_start:b_end]
        
        for i in range(b_len):
            col_idx = b_start + i
            w_col = W_compressed[:, col_idx].copy()
            
            diag_val = H_inv[col_idx, col_idx]
            if diag_val == 0:
                continue
                
            row_sort_idx = np.argsort(np.abs(W[:, col_idx]))
            prune_mask = np.zeros(out_dim, dtype=bool)
            if k > 0:
                prune_mask[row_sort_idx[:k]] = True
                
            w_quant = quantize_tensor(w_col, num_bits)
            w_new = np.where(prune_mask, 0.0, w_quant)
            
            err = (w_col - w_new) / diag_val
            W_compressed[:, col_idx] = w_new
            
            rem_len = in_dim - (col_idx + 1)
            if rem_len > 0:
                H_col = H_inv[col_idx, col_idx + 1:]
                W_compressed[:, col_idx + 1:] -= np.outer(err, H_col)

    return W_compressed
