import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        ([np.array([3.0, 4.0])], 2.0),
        ([rng.normal(size=(10, 10)), rng.normal(size=(5,))], 1.0),
        ([rng.normal(size=(2, 2)) * 0.1, rng.normal(size=(3,)) * 0.1], 10.0),
        ([np.zeros((3, 3)), np.zeros(2)], 1.0),
        ([rng.normal(size=(3, 4, 5)), rng.normal(size=(2,))], 2.5),
    ]
    
    max_err = 0.0
    for grads, max_norm in cases:
        # ORACLE: Compute using exact formula
        total_norm_sq = 0.0
        for g in grads:
            total_norm_sq += np.sum(g**2)
        total_norm = np.sqrt(total_norm_sq)
        
        coef = min(1.0, max_norm / (total_norm + 1e-6))
        expected = [g * coef for g in grads]
        
        # Evaluate sol
        grads_copy = [g.copy() for g in grads]
        try:
            ans = sol.clip_global_norm(grads_copy, max_norm)
            if not isinstance(ans, list) or len(ans) != len(expected):
                return {"max_abs_err": 1e9}
                
            for a, e in zip(ans, expected):
                if a.shape != e.shape:
                    return {"max_abs_err": 1e9}
                max_err = max(max_err, np.max(np.abs(a - e)))
                
        except Exception:
            return {"max_abs_err": float('inf')}
            
    return {"max_abs_err": float(max_err)}
