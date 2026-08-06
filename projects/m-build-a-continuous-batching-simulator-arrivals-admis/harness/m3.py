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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_admission_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
        
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
        
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
        
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0
    
    import cbsim.engine as eng
    good_cont = eng.simulate_continuous
    
    def bad_continuous(requests, max_batch_size):
        pending = list(requests)
        running = []
        tick = 0
        log = []
        
        while pending or running:
            available = [r for r in pending if r.arrival <= tick]
            if not available and not running and pending:
                tick = pending[0].arrival
                available = [r for r in pending if r.arrival <= tick]
                
            admit_count = min(1, max_batch_size - len(running))
            for r in available[:admit_count]:
                pending.remove(r)
                running.append(r.decode_len)
                
            log.append(len(running))
            running = [rem - 1 for rem in running if rem > 1]
            tick += 1
            
        return tick, log
        
    eng.simulate_continuous = bad_continuous
    
    try:
        out["catches_admission_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        eng.simulate_continuous = good_cont
        
    return out
