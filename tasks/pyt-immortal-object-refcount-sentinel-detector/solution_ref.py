import ctypes


def _objects():
    objs = [None, True, False]
    objs.extend(range(-5, 257))
    objs.extend([str(x) for x in ["alpha", "beta", "gamma", "delta"]])
    objs.append(object())
    return objs


def _raw_refcnt(obj):
    return ctypes.c_ssize_t.from_address(id(obj)).value


def immortal_refcount_sentinel_detector():
    objs = _objects()
    counts = [_raw_refcnt(obj) for obj in objs]
    sentinel = max(counts)
    return [count == sentinel for count in counts]
