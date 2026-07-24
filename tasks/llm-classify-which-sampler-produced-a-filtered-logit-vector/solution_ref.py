import numpy as np

def classify_sampler(orig_logits: np.ndarray, filtered_logits: np.ndarray) -> str:
    kept = ~np.isneginf(filtered_logits)
    counts = np.sum(kept, axis=-1)
    
    if np.all(counts == 1):
        return "greedy"
        
    if np.all(counts == counts[0]):
        return "top-k"
        
    probs = np.exp(orig_logits - np.max(orig_logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    
    # Check top-p
    max_C_prime = 0.0
    min_C = 1.0
    for i in range(orig_logits.shape[0]):
        sorted_p = np.sort(probs[i])[::-1]
        k_i = counts[i]
        C_i = np.sum(sorted_p[:k_i])
        C_prime = np.sum(sorted_p[:k_i-1]) if k_i > 1 else 0.0
        max_C_prime = max(max_C_prime, C_prime)
        min_C = min(min_C, C_i)
        
    if max_C_prime + 1e-5 < min_C:
        return "top-p"
        
    # Check min-p
    max_R = 0.0
    min_K = 1.0
    for i in range(orig_logits.shape[0]):
        max_p = np.max(probs[i])
        K_i = np.min(probs[i, kept[i]]) / max_p
        R_i = np.max(probs[i, ~kept[i]]) / max_p if np.any(~kept[i]) else 0.0
        max_R = max(max_R, R_i)
        min_K = min(min_K, K_i)
        
    if max_R + 1e-5 < min_K:
        return "min-p"
        
    return "unknown"
