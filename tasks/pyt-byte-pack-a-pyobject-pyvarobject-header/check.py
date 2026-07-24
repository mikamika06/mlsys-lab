import ctypes
import struct


def _read_u64(addr):
    return int(ctypes.c_uint64.from_address(addr).value)


def _oracle_pack(obj, type_ids, is_var_object):
    base = id(obj)
    refcnt = _read_u64(base)
    type_id = type_ids[type(obj)]
    size = _read_u64(base + 16) if is_var_object else 0
    return struct.pack("<QQQ", refcnt, type_id, size)


def grade(sol, fx) -> dict:
    cases = [
        ([1, 2, 3], {list: 11}, True),
        ("hello", {str: 12}, True),
        ({1: "a", 2: "b"}, {dict: 13}, False),
        (42, {int: 14}, False),
        (bytearray(b"abc"), {bytearray: 15}, True),
    ]

    total_same = 0
    total_bytes = 0

    for obj, type_ids, is_var in cases:
        expected = _oracle_pack(obj, type_ids, is_var)
        try:
            got = sol.pack_pyobject_header(obj, type_ids, is_var)
        except Exception:
            return {"byte_exact_fraction": 0.0}

        if not isinstance(got, bytes):
            return {"byte_exact_fraction": 0.0}

        same = sum(a == b for a, b in zip(got, expected))
        total_same += same
        total_bytes += max(len(got), len(expected))

    if total_bytes == 0:
        return {"byte_exact_fraction": 0.0}
    return {"byte_exact_fraction": total_same / total_bytes}
