import math
import numpy as np

def cross_entropy_loss(logits, targets, mask=None):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    shape = logits.shape
    outer_shape = shape[:-1]
    K = shape[-1]

    out = np.empty(outer_shape, dtype=np.float32)

    for idx in np.ndindex(outer_shape):
        t = targets[idx]
        
        max_v = logits[idx + (0,)]
        for k in range(1, K):
            v = logits[idx + (k,)]
            if v > max_v:
                max_v = v

        sum_exp = 0.0
        for k in range(K):
            sum_exp += math.exp(logits[idx + (k,)] - max_v)

        prob_t = math.exp(logits[idx + (t,)] - max_v) / sum_exp
        log_prob_t = math.log(prob_t + 1e-12)
        out[idx] = -log_prob_t

    if mask is not None:
        batch_shape = outer_shape[:-1]
        L = outer_shape[-1]
        res = np.empty(batch_shape, dtype=np.float32)

        for b_idx in np.ndindex(batch_shape):
            ce_sum = 0.0
            denom = 0
            for i in range(L):
                m = mask[b_idx + (i,)]
                if m:
                    ce_sum += out[b_idx + (i,)]
                    denom += 1
            if denom > 0:
                res[b_idx] = ce_sum / denom
            else:
                res[b_idx] = 0.0
        return res
    else:
        batch_shape = outer_shape[:-1]
        L = outer_shape[-1]
        res = np.empty(batch_shape, dtype=np.float32)

        for b_idx in np.ndindex(batch_shape):
            ce_sum = 0.0
            for i in range(L):
                ce_sum += out[b_idx + (i,)]
            res[b_idx] = ce_sum / L
        return res
