import ref
import random

def check(workdir):
    out = {"hist_err": 1.0}
    
    try:
        import cbsim.metrics as met
    except ImportError:
        out["_note"] = "Could not import cbsim.metrics"
        return out
        
    random.seed(99)
    log = [random.randint(0, 5) for _ in range(100)]
    
    try:
        r_hist = ref.occupancy_histogram(log, 4)
        l_hist = met.occupancy_histogram(log, 4)
        
        if not isinstance(l_hist, list) or len(r_hist) != len(l_hist):
            out["_note"] = f"Expected list of length {len(r_hist)}, got length {len(l_hist) if hasattr(l_hist, '__len__') else 'unknown'}"
            return out
            
        diff = sum(abs(a - b) for a, b in zip(r_hist, l_hist))
        out["hist_err"] = float(diff) / (sum(r_hist) or 1.0)
    except Exception as e:
        out["_note"] = f"occupancy_histogram failed: {e}"
        
    return out
