import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_undercover": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    
    if not os.path.isfile(path):
        return out
        
    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out
        
    if first is None:
        return out
        
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0
    
    import profiler.analyzer as pa
    good_hot_op = pa.hot_op_cover
    
    def broken_hot_op_cover(events, threshold=0.8):
        ts = pa.detect_warmup(events)
        steady = [e for e in events if e.get("ts", 0) >= ts]
        op_durs = {}
        for e in steady:
            if e.get("cat") == "Node" and "args" in e and "op_name" in e["args"]:
                op = e["args"]["op_name"]
                op_durs[op] = op_durs.get(op, 0) + e["dur"]
        
        total = sum(op_durs.values())
        target = threshold * total
        sorted_ops = sorted(op_durs.items(), key=lambda x: (-x[1], x[0]))
        
        cover = set()
        accum = 0
        for op, dur in sorted_ops:
            if accum + dur >= target and len(cover) > 0:
                break
            cover.add(op)
            accum += dur
        return cover
        
    pa.hot_op_cover = broken_hot_op_cover
    sys.modules["profiler.analyzer"].hot_op_cover = broken_hot_op_cover
    
    try:
        out["catches_undercover"] = 0.0 if _survives(path) else 1.0
    finally:
        pa.hot_op_cover = good_hot_op
        sys.modules["profiler.analyzer"].hot_op_cover = good_hot_op
        
    return out
