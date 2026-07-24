import numpy as np

def _ref_oracle(P, eps):
    return P > eps

def grade(sol, fx) -> dict:
    np.random.seed(2025)
    # Test 1: causal mask from random logits
    n = 8
    logits = np.random.randn(n, n).astype(np.float64)
    # build causal mask: lower triangular (including diagonal)
    mask = np.tril(np.ones((n, n), dtype=np.float64))
    masked_logits = np.where(mask == 0, -np.inf, logits)
    # softmax row-wise
    exp_logits = np.exp(masked_logits - np.max(masked_logits, axis=1, keepdims=True))
    P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    true_mask = _ref_oracle(P, 1e-12)
    try:
        pred_mask = sol.recover_mask(P, 1e-12)
    except Exception:
        return {"exact_match": 0.0}
    if not np.all(pred_mask == true_mask):
        return {"exact_match": 0.0}

    # Test 2: random sparse mask with large block of zeros
    n2 = 6
    logits2 = np.random.randn(n2, n2).astype(np.float64)
    mask2 = np.random.randint(0, 2, size=(n2, n2)).astype(np.float64)
    masked_logits2 = np.where(mask2 == 0, -np.inf, logits2)
    exp_logits2 = np.exp(masked_logits2 - np.max(masked_logits2, axis=1, keepdims=True))
    P2 = exp_logits2 / np.sum(exp_logits2, axis=1, keepdims=True)
    true_mask2 = _ref_oracle(P2, 1e-12)
    try:
        pred_mask2 = sol.recover_mask(P2, 1e-12)
    except Exception:
        return {"exact_match": 0.0}
    if not np.all(pred_mask2 == true_mask2):
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
