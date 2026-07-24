import numpy as np

def _reference(logits_list):
    thresh = np.log(np.finfo(np.float32).max)
    return [log.max() > thresh for log in logits_list]

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []
    # 1. small values no overflow
    cases.append(rng.uniform(-10, 10, size=5))
    # 2. near threshold but below
    thresh = np.log(np.finfo(np.float32).max)
    cases.append(np.linspace(thresh-5, thresh-0.1, num=7))
    # 3. above threshold
    cases.append(np.linspace(thresh+0.1, thresh+10, num=8))
    # 4. mix of negative and large positive
    arr = rng.uniform(-50, 100, size=10)
    cases.append(arr)
    # 5. all equal to threshold
    cases.append(np.full(6, thresh))
    logits_list = [np.array(c) for c in cases]
    try:
        got = sol.predict_overflow(logits_list)
    except Exception:
        return {"exact_match": 0.0}
    try:
        got_list = [bool(x) for x in got]
    except TypeError:
        return {"exact_match": 0.0}
    ref = _reference(logits_list)
    ok = int(got_list == ref)
    return {"exact_match": float(ok)}
