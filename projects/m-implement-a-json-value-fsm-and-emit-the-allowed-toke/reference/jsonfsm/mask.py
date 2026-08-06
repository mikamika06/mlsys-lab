import numpy as np

def compute_mask(logits, allowed_indices):
    mask = np.full_like(logits, -float("inf"))
    mask[allowed_indices] = 0.0
    return logits + mask

def verify_equivalence(original_logits, masked_logits, allowed_indices):
    allowed_orig = original_logits[allowed_indices]
    allowed_masked = masked_logits[allowed_indices]
    if len(allowed_orig) < 2:
        return True
    order_orig = np.argsort(allowed_orig)
    order_masked = np.argsort(allowed_masked)
    return np.array_equal(order_orig, order_masked)
