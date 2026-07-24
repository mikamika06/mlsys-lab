import numpy as np

def _ref(mask: np.ndarray) -> str:
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

def grade(sol, fx) -> dict:
    cases = []
    
    # 1. Bidirectional
    cases.append(np.ones((8, 8), dtype=bool))
    
    # 2. Causal
    cases.append(np.tril(np.ones((10, 10), dtype=bool)))
    
    # 3. Window (w=3)
    n = 12
    cases.append(np.tril(np.ones((n, n), dtype=bool)) & np.triu(np.ones((n, n), dtype=bool), -3))
    
    # 4. Window (w=1)
    n = 6
    cases.append(np.tril(np.ones((n, n), dtype=bool)) & np.triu(np.ones((n, n), dtype=bool), -1))
    
    # 5. Prefix-LM (P=4)
    n = 9
    pm = np.tril(np.ones((n, n), dtype=bool))
    pm[:4, :4] = True
    cases.append(pm)
    
    # 6. Prefix-LM (P=1) (Note: P=1 prefix is identical to causal! We ensure P > 1 to be unique if needed,
    # but P=1 prefix is technically causal. Let's use P=2.)
    n = 7
    pm2 = np.tril(np.ones((n, n), dtype=bool))
    pm2[:2, :2] = True
    cases.append(pm2)

    ref_ans = [_ref(c) for c in cases]
    
    try:
        got_ans = sol.classify_masks(cases)
    except Exception:
        return {"exact_match": 0.0}
        
    if got_ans == ref_ans:
        return {"exact_match": 1.0}
    else:
        return {"exact_match": 0.0}
