import numpy as np

def classify_masks(masks: list[np.ndarray]) -> list[str]:
    def classify(mask: np.ndarray) -> str:
        n = mask.shape[0]
        
        if np.all(mask):
            return "bidirectional"
            
        causal = np.tril(np.ones((n, n), dtype=bool))
        if np.array_equal(mask, causal):
            return "causal"
            
        for w in range(1, n - 1):
            window_mask = np.tril(np.ones((n, n), dtype=bool)) & np.triu(np.ones((n, n), dtype=bool), -w)
            if np.array_equal(mask, window_mask):
                return "window"
                
        for p in range(1, n):
            prefix_mask = np.tril(np.ones((n, n), dtype=bool))
            prefix_mask[:p, :p] = True
            if np.array_equal(mask, prefix_mask):
                return "prefix-lm"
                
        return "unknown"
        
    return [classify(m) for m in masks]
