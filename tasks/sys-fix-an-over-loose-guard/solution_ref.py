def guard_ok(cached_meta: dict, new_meta: dict) -> bool:
    """
    Decide whether the compiled graph traced for `cached_meta` may be
    safely reused for a new call described by `new_meta` (i.e. whether
    NO recompilation is needed).

    cached_meta / new_meta: dicts with keys
        "shape": tuple[int, ...]
        "dtype": str  (e.g. "float64", "int32")

    Reuse is only safe when shape AND dtype are both exactly equal.
    """
    return (cached_meta["shape"] == new_meta["shape"]
            and cached_meta["dtype"] == new_meta["dtype"])
