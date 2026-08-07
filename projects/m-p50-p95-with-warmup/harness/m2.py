import ref
import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from measure import benchmark
    except ImportError:
        return {"rel_err": 1000.0, "_note": "measure.py missing or not importable"}

    clock, fn = ref.make_clock_and_fn(ref.DELAYS_M2)
    want = ref.benchmark(fn, 3, 10, reject_outliers=True, clock=clock)

    clock, fn = ref.make_clock_and_fn(ref.DELAYS_M2)
    try:
        got = benchmark(fn, 3, 10, reject_outliers=True, clock=clock)
    except Exception as e:
        return {"rel_err": 1000.0, "_note": f"crash: {e}"}

    if not isinstance(got, dict) or not all(k in want for k in got) or not all(k in got for k in want):
        return {"rel_err": 1000.0, "_note": f"missing keys, expected {list(want.keys())}"}

    err = sum(abs(got[k] - want[k]) / (abs(want[k]) + 1e-9) for k in want)

    clock, fn = ref.make_clock_and_fn(ref.DELAYS_M2)
    want_no_reject = ref.benchmark(fn, 3, 10, reject_outliers=False, clock=clock)
    err_from_no_reject = sum(abs(got[k] - want_no_reject[k]) / (abs(want_no_reject[k]) + 1e-9) for k in want_no_reject)
    if err_from_no_reject < 1e-3:
        return {"rel_err": 1000.0, "_note": "Output matches the no-rejection case. Did you implement reject_outliers?"}

    return {"rel_err": float(err)}
