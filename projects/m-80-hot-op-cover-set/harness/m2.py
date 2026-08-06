import sys
import math
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    import profiler.analyzer as pa
    
    out = {"attrib_match": 1.0}
    
    for trace_pair in ref.TRACES:
        got = pa.attribute_speedup(trace_pair["before"], trace_pair["after"])
        want = ref.attribute_speedup(trace_pair["before"], trace_pair["after"])
        
        if got.keys() != want.keys():
            out["attrib_match"] = 0.0
            break
            
        for k in want:
            if not math.isclose(got[k], want[k], rel_tol=1e-5, abs_tol=1e-5):
                out["attrib_match"] = 0.0
                break
                
    return out
