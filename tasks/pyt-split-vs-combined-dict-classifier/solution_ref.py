import ctypes


class _PyDictObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ma_used", ctypes.c_ssize_t),
        ("ma_version_tag", ctypes.c_uint64),
        ("ma_keys", ctypes.c_void_p),
        ("ma_values", ctypes.c_void_p),
    ]


def is_split_dict(d: dict) -> bool:
    """
    Return True if `d` is currently a CPython split-table dict (its
    values live in a separate `ma_values` array shared/keyed against a
    `ma_keys` table it does not own outright), False if it is a normal
    combined dict.

    Implemented by reading the live PyDictObject header: CPython's own
    source treats `ma_values != NULL` as the definition of "has a split
    table" (see `_PyDict_HasSplitTable` in Objects/dictobject.c).
    """
    raw = _PyDictObject.from_address(id(d))
    if raw.ma_used != len(d):
        raise RuntimeError("PyDictObject layout mismatch on this interpreter")
    return raw.ma_values is not None
