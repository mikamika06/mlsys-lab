"""Grader for `pyt-reimplement-getattribute-resolution-order`.

Two independent gates:

  * exact_match -- oracle is REAL CPython attribute access (`getattr` on
    real instances of real classes with real descriptors), never a
    hand-derived table. Whatever the real interpreter actually resolves
    each (obj, name) pair to is the ground truth.
  * no_shortcut -- inspects the submitted function's bytecode (via its
    `__code__.co_names`, recursively through any nested code objects)
    to confirm it never references `getattr`, `super`, or
    `__getattribute__` -- the three ways to trivially delegate the
    whole problem back to the real attribute-lookup machinery instead
    of reimplementing it.
"""
from __future__ import annotations

import types


# ---- fixture classes: one per precedence tier ----

class _DataDescriptor:
    def __init__(self, value):
        self.value = value

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.value

    def __set__(self, obj, value):
        self.value = value


class _NonDataDescriptor:
    def __init__(self, value):
        self.value = value

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.value


class _HasBoth:                      # data descriptor + plain class attr + instance attr
    dd = _DataDescriptor("HasBoth.dd")
    ndd = _NonDataDescriptor("HasBoth.ndd")
    plain = "HasBoth.plain"

    def __init__(self):
        self.inst_only = "inst_only"


class _NddShadowedByInstance:        # non-data descriptor LOSES to instance dict
    ndd = _NonDataDescriptor("Shadow.ndd")

    def __init__(self):
        self.__dict__["ndd"] = "Shadow.inst.ndd"


class _DdBeatsRawInstanceEntry:      # data descriptor WINS over a raw instance-dict entry
    dd = _DataDescriptor("Beats.dd")

    def __init__(self):
        self.__dict__["dd"] = "Beats.inst.dd (must be ignored)"


class _GetattrFallback:
    def __getattr__(self, name):
        return f"fallback:{name}"


class _TotalMiss:
    pass


class _Base1:
    z = "base1"


class _Base2:
    z = "base2"


class _Diamond(_Base1, _Base2):      # MRO order matters, not declaration order
    pass


class _PlainOverride:                # ordinary instance attr shadowing a plain class attr
    a = 10

    def __init__(self):
        self.a = 5


def _cases():
    return [
        (_HasBoth(), "dd"),
        (_HasBoth(), "ndd"),
        (_HasBoth(), "plain"),
        (_HasBoth(), "inst_only"),
        (_NddShadowedByInstance(), "ndd"),
        (_DdBeatsRawInstanceEntry(), "dd"),
        (_GetattrFallback(), "whatever"),
        (_TotalMiss(), "missing"),
        (_Diamond(), "z"),
        (_PlainOverride(), "a"),
    ]


_FORBIDDEN = {"getattr", "super", "__getattribute__"}


def _uses_forbidden(fn) -> bool:
    def walk(code) -> bool:
        if any(n in _FORBIDDEN for n in code.co_names):
            return True
        return any(
            isinstance(c, types.CodeType) and walk(c) for c in code.co_consts
        )
    try:
        return walk(fn.__code__)
    except AttributeError:
        return True   # not even a plain function -- treat as unsafe


def grade(sol, fx) -> dict:
    try:
        no_shortcut = 0.0 if _uses_forbidden(sol.resolve) else 1.0
    except Exception:
        return {"exact_match": 0.0, "no_shortcut": 0.0}

    hits = 0
    cases = _cases()
    for obj, name in cases:
        try:
            expected = getattr(obj, name)
            exp_exc = None
        except AttributeError:
            expected = None
            exp_exc = "AttributeError"

        try:
            got = sol.resolve(obj, name)
            got_exc = None
        except AttributeError:
            got = None
            got_exc = "AttributeError"
        except Exception:
            got = None
            got_exc = "OTHER"

        if exp_exc == got_exc and (exp_exc is not None or got == expected):
            hits += 1

    return {
        "exact_match": hits / len(cases),
        "no_shortcut": no_shortcut,
    }
