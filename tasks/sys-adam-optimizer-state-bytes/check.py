def oracle_adam_bytes(params, mixed):
    bytes_m = params * 4
    bytes_v = params * 4
    bytes_master = params * 4 if mixed else 0
    return bytes_m + bytes_v + bytes_master

def grade(sol, fx):
    cases = [
        (1000, False),
        (1000, True),
        (5000000, False),
        (5000000, True),
        (0, False)
    ]
    
    for num_params, mixed in cases:
        ref_ans = oracle_adam_bytes(num_params, mixed)
        try:
            ans = sol.adam_optimizer_state_bytes(num_params, mixed)
            
            # Handle edge case for 0 params
            if ref_ans == 0:
                if ans != 0:
                    return {"size_ratio": float('inf')}
                continue
                
            ratio = ans / ref_ans
            if abs(ratio - 1.0) > 1e-9:
                return {"size_ratio": float(ratio)}
        except Exception:
            return {"size_ratio": 0.0}
            
    return {"size_ratio": 1.0}
