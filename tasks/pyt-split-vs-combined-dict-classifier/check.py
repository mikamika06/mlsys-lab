import ctypes


class _PyDictObject(ctypes.Structure):
    """Layout of CPython's PyDictObject (ma_used/ma_keys/ma_values tail)."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ma_used", ctypes.c_ssize_t),
        ("ma_version_tag", ctypes.c_uint64),
        ("ma_keys", ctypes.c_void_p),
        ("ma_values", ctypes.c_void_p),
    ]


def _oracle_is_split(d: dict) -> bool:
    """Ground truth: read the live PyDictObject and check ma_values.

    CPython itself defines "split table" as `ma_values != NULL` (this is
    exactly the condition `_PyDict_HasSplitTable` checks in dictobject.c).
    A combined dict stores keys+hashes+values together in `ma_keys` and
    `ma_values` is NULL; a split dict stores only values in `ma_values`
    and shares its key table with other dicts (e.g. instances of the
    same class).
    """
    raw = _PyDictObject.from_address(id(d))
    # Sanity check that the struct offsets line up with this build/arch;
    # if they didn't, ma_used would not equal the real dict length.
    if raw.ma_used != len(d):
        raise RuntimeError("PyDictObject layout mismatch on this interpreter")
    return raw.ma_values is not None


def _build_cases():
    """Construct labeled (dict, oracle_label) pairs covering distinct
    real CPython code paths. Labels come from live introspection above,
    never from a written-down expectation."""
    cases = []

    class PointA:
        pass

    p1 = PointA()
    p1.x = 1
    p1.y = 2
    cases.append(("instance dict, fresh attrs", p1.__dict__))

    p2 = PointA()
    p2.x = 10
    p2.y = 20
    cases.append(("instance dict, second instance same order", p2.__dict__))

    cases.append(("plain dict literal", {"x": 1, "y": 2}))
    cases.append(("dict() with kwargs", dict(x=1, y=2)))
    cases.append(
        ("dict comprehension", {k: v for k, v in [("x", 1), ("y", 2)]})
    )

    class PointB:
        pass

    p3 = PointB()
    p3.x = 1
    p3.y = 2
    p3.z = 3
    del p3.y
    cases.append(("instance dict after `del obj.attr`", p3.__dict__))

    class PointC:
        pass

    p4 = PointC()
    p4.a = 1
    p4.b = 2
    cases.append(("dict(instance.__dict__) copy", dict(p4.__dict__)))

    class PointD:
        pass

    p5 = PointD()
    p5.a = 1
    p5.__dict__[42] = "nonstring-key"
    cases.append(("instance dict, non-string key inserted", p5.__dict__))

    class PointE:
        pass

    p6 = PointE()
    cases.append(("instance dict, no attrs set yet", p6.__dict__))

    class PointF:
        pass

    p7 = PointF()
    p7.a = 1
    p7.b = 2
    cases.append(("vars(instance)", vars(p7)))

    class PointG:
        pass

    p8 = PointG()
    p8.__dict__.update({"a": 1, "b": 2})
    cases.append(("instance dict built via .update()", p8.__dict__))

    class DictSub(dict):
        pass

    cases.append(("dict subclass instance", DictSub(x=1, y=2)))

    return cases


def grade(sol, fx) -> dict:
    cases = _build_cases()
    total = len(cases)
    correct = 0

    for _name, d in cases:
        try:
            expected = _oracle_is_split(d)
        except Exception:
            # Oracle itself must always succeed; if it can't, this
            # environment can't grade the task at all.
            return {"exact_match": 0.0}

        try:
            got = sol.is_split_dict(d)
        except Exception:
            continue

        if isinstance(got, bool) and got == expected:
            correct += 1

    return {"exact_match": correct / total if total else 0.0}
