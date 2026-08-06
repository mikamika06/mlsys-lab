import sys
import os

def check(workdir):
    sys.path.insert(0, os.path.join(workdir))
    import ref
    from compat.suite import run_compatibility_suite
    
    out = {"total_executed": 0.0, "responses_valid": 0.0}
    server = ref.MockServer()
    
    try:
        results = run_compatibility_suite(server, ref.SHAPES, ref.make_sample_request)
        out["total_executed"] = float(len(results))
        valid_cnt = sum(1 for v in results.values() if v.get("valid") is True)
        out["responses_valid"] = float(valid_cnt)
    except Exception as e:
        out["_note"] = f"Suite run failed: {type(e).__name__}: {str(e)}"
        
    return out
