import ref


def check(workdir):
    from mps_bench.core import inspect_device_flags
    out = {"status_matched": 0.0}
    correct = 0
    total = len(ref.DEVICE_TEST_CASES)
    for case in ref.DEVICE_TEST_CASES:
        want = {
            "is_built": bool(case["is_built"]),
            "is_available": bool(case["is_available"]),
            "valid_state": not (case["is_available"] and not case["is_built"])
        }
        try:
            got = inspect_device_flags(case["is_built"], case["is_available"])
            if got == want:
                correct += 1
        except Exception:
            pass
    if correct == total:
        out["status_matched"] = 1.0
    return out
