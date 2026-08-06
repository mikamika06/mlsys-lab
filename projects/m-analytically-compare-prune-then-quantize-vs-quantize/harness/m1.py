import ref
import numpy as np


def check(workdir):
    from order_effect.analytic import compare_order_error
    cases = ref.generate_test_cases()
    
    matched = 0
    max_err = 0.0
    
    for W, X, sparsity, num_bits in cases:
        got = compare_order_error(W, X, sparsity, num_bits)
        
        Y = W @ X
        W_p = np.copy(W)
        for r in range(W.shape[0]):
            k = int(np.round(W.shape[1] * sparsity))
            if k > 0:
                idx = np.argsort(np.abs(W_p[r]))[:k]
                W_p[r, idx] = 0.0
                
        qmax = (2 ** num_bits) - 1
        w_min, w_max = np.min(W_p), np.max(W_p)
        scale = (w_max - w_min) / qmax if w_max != w_min else 1.0
        zp = np.round(-w_min / scale) if w_max != w_min else 0.0
        W_ptq = (np.clip(np.round(W_p / scale + zp), 0, qmax) - zp) * scale
        Y_ptq = W_ptq @ X
        want_ptq = float(np.mean((Y - Y_ptq) ** 2))
        
        w_min2, w_max2 = np.min(W), np.max(W)
        scale2 = (w_max2 - w_min2) / qmax if w_max2 != w_min2 else 1.0
        zp2 = np.round(-w_min2 / scale2) if w_max2 != w_min2 else 0.0
        W_q = (np.clip(np.round(W / scale2 + zp2), 0, qmax) - zp2) * scale2
        W_qtp = np.copy(W_q)
        for r in range(W.shape[0]):
            k = int(np.round(W.shape[1] * sparsity))
            if k > 0:
                idx = np.argsort(np.abs(W_qtp[r]))[:k]
                W_qtp[r, idx] = 0.0
        Y_qtp = W_qtp @ X
        want_qtp = float(np.mean((Y - Y_qtp) ** 2))
        
        err_ptq = abs(got["ptq_mse"] - want_ptq)
        err_qtp = abs(got["qtp_mse"] - want_qtp)
        err = max(err_ptq, err_qtp)
        if err > max_err:
            max_err = err
            
        if err < 1e-5:
            matched += 1

    return {
        "cases_matched": float(matched),
        "mse": float(max_err)
    }
