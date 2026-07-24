import ctypes
import struct
import sys


def _oracle_header(obj):
    refcnt = sys.getrefcount(obj) - 1
    ptr = id(type(obj))
    word = ctypes.sizeof(ctypes.c_void_p)
    return struct.pack("@q", refcnt) + struct.pack("@Q", ptr) if word == 8 else b""


def grade(sol, fx) -> dict:
    cases = [
        [],
        {},
        {"a": 1, "b": 2},
        (1, 2, 3),
        object(),
        None,
    ]

    try:
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            return {"byte_exact_fraction": 0.0}
        if not hasattr(sol, "pack_object_header"):
            return {"byte_exact_fraction": 0.0}
    except Exception:
        return {"byte_exact_fraction": 0.0}

    total = 0
    same = 0
    for obj in cases:
        expected = _oracle_header(obj)
        try:
            got = bytes(sol.pack_object_header(obj))
        except Exception:
            continue
        total += len(expected)
        same += sum(a == b for a, b in zip(got, expected))
        if len(got) != len(expected):
            continue

    return {"byte_exact_fraction": same / total if total else 0.0}
