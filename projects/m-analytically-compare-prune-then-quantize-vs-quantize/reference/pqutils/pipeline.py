import numpy as np


def run_joint_pipeline(weights, hessian, sparsity, q_bits, order):
    w = np.array(weights, dtype=np.float64)
    h = np.array(hessian, dtype=np.float64)
    scale = (2.0 ** (q_bits - 1) - 1.0)
    if scale <= 0:
        scale = 1.0

    if order == "prune_then_quantize":
        flat = np.abs(w.flatten())
        thresh = np.percentile(flat, sparsity * 100.0)
        mask = (flat >= thresh).reshape(w.shape)
        w_p = np.where(mask, w, 0.0)

        inv_h = np.linalg.inv(h + 1e-5 * np.eye(h.shape[0]))
        w_sparse_sim = w_p @ inv_h @ h

        w_final = np.round(w_sparse_sim * scale) / scale
        return w_final
    elif order == "quantize_then_prune":
        w_q = np.round(w * scale) / scale
        flat = np.abs(w_q.flatten())
        thresh = np.percentile(flat, sparsity * 100.0)
        mask = (flat >= thresh).reshape(w_q.shape)
        w_final = np.where(mask, w_q, 0.0)
        return w_final
    else:
        raise ValueError(f"Unknown order: {order}")
