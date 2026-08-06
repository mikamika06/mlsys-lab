import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def check(workdir):
    out = {"exact_match": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out
    
    try:
        first = _run(path)
    except Exception:
        return out
    
    if first is None:
        return out
    
    import hf_attn.suite as suite
    import numpy as np
    
    good_assert = suite.assert_parity_on_valid
    
    def bad_assert(q, k, v, mask, ref_fn, test_fn):
        return float(np.max(np.abs(ref_fn(q, k, v, mask) - test_fn(q, k, v, mask))))
    
    suite.assert_parity_on_valid = bad_assert
    import hf_attn
    hf_attn.suite.assert_parity_on_valid = bad_assert
    
    try:
        survives = False
        try:
            survives = _run(path) is True
        except Exception:
            pass
        if not survives:
            out["exact_match"] = 1.0
    finally:
        suite.assert_parity_on_valid = good_assert
        hf_attn.suite.assert_parity_on_valid = good_assert
        
    return out
