def guard_ok(cached_meta: dict, new_meta: dict) -> bool:
    """
    Decide whether the compiled graph traced for `cached_meta` may be
    safely reused for a new call described by `new_meta` (i.e. whether
    NO recompilation is needed).

    cached_meta / new_meta: dicts with keys
        "shape": tuple[int, ...]
        "dtype": str  (e.g. "float64", "int32")

    BUG: this checks dtype correctly but, instead of comparing the exact
    shape tuple, only compares total element count -- so two different
    shapes with the same number of elements (e.g. (2,6) and (3,4)) wrongly
    pass the guard and a stale graph gets reused. Fix it.
    """
    if cached_meta["dtype"] != new_meta["dtype"]:
        return False

    cached_numel = 1
    for d in cached_meta["shape"]:
        cached_numel *= d
    new_numel = 1
    for d in new_meta["shape"]:
        new_numel *= d

    return cached_numel == new_numel
