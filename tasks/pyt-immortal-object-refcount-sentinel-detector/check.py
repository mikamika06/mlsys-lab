import ctypes


def _objects():
    objs = [None, True, False]
    objs.extend(range(-5, 257))
    objs.extend([str(x) for x in ["alpha", "beta", "gamma", "delta"]])
    objs.append(object())
    return objs


def _raw_refcnt(obj):
    return ctypes.c_ssize_t.from_address(id(obj)).value


def _ref():
    objs = _objects()
    counts = [_raw_refcnt(obj) for obj in objs]
    sentinel = max(counts)
    return [count == sentinel for count in counts]


def grade(sol, fx) -> dict:
    try:
        got = sol.immortal_refcount_sentinel_detector()
    except Exception:
        return {"exact_match": 0.0}
    expected = _ref()
    return {"exact_match": 1.0 if list(got) == expected else 0.0}
