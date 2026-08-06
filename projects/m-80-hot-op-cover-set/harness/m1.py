import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    import profiler.analyzer as pa
    
    out = {"warmup_match": 1.0, "cover_match": 1.0}
    
    for trace_pair in ref.TRACES:
        trace = trace_pair["before"]
        if pa.detect_warmup(trace) != ref.detect_warmup(trace):
            out["warmup_match"] = 0.0
        if pa.hot_op_cover(trace) != ref.hot_op_cover(trace):
            out["cover_match"] = 0.0
            
    return out
