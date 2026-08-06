import numpy as np

def top_k_filter(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a boolean mask of the indices belonging to the top‑k logits and
    the filtered logits where all other entries are set to -inf.
    """
    n = len(logits)
    indices = []
    for i in range(n):
        indices.append(i)
        
    for i in range(n):
        for j in range(0, n - i - 1):
            if -logits[indices[j]] > -logits[indices[j + 1]]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
                
    mask = []
    for _ in range(n):
        mask.append(False)
        
    for idx in indices[:k]:
        mask[idx] = True
        
    filtered = []
    for i in range(n):
        if mask[i]:
            filtered.append(logits[i])
        else:
            filtered.append(float('-inf'))
            
    return np.array(mask, dtype=bool), np.array(filtered, dtype=logits.dtype)
