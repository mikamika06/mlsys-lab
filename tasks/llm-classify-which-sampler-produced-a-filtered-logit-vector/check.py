import numpy as np

def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def apply_greedy(logits):
    out = np.full_like(logits, -np.inf)
    idx = np.argmax(logits, axis=-1)
    out[np.arange(logits.shape[0]), idx] = logits[np.arange(logits.shape[0]), idx]
    return out

def apply_top_k(logits, k):
    out = np.copy(logits)
    thresholds = np.sort(logits, axis=-1)[:, -k]
    out[logits < thresholds[:, None]] = -np.inf
    return out

def apply_top_p(logits, p):
    probs = _softmax(logits)
    sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
    
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
    sorted_indices_to_remove[..., 0] = False
    
    out = np.copy(logits)
    for i in range(logits.shape[0]):
        out[i, sorted_indices[i, sorted_indices_to_remove[i]]] = -np.inf
    return out

def apply_min_p(logits, min_p):
    probs = _softmax(logits)
    max_probs = np.max(probs, axis=-1, keepdims=True)
    out = np.copy(logits)
    out[probs < max_probs * min_p] = -np.inf
    return out

def grade(sol, fx) -> dict:
    np.random.seed(42)
    orig = np.random.randn(32, 256) * 3
    
    cases = [
        (orig, apply_greedy(orig), "greedy"),
        (orig, apply_top_k(orig, 15), "top-k"),
        (orig, apply_top_p(orig, 0.9), "top-p"),
        (orig, apply_min_p(orig, 0.05), "min-p")
    ]
    
    exact_match = 1.0
    for o, filt, ref_label in cases:
        try:
            got_label = sol.classify_sampler(o, filt)
        except Exception:
            return {"exact_match": 0.0}
            
        if got_label != ref_label:
            exact_match = 0.0
            break
            
    return {"exact_match": exact_match}
