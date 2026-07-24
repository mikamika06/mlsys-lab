import ctypes
import struct


def _read_u64(addr):
    return int(ctypes.c_uint64.from_address(addr).value)


def pack_pyobject_header(obj, type_ids, is_var_object):
    refcnt = _read_u64(id(obj))
    type_id = type_ids[type(obj)]
    size = _read_u64(id(obj) + 16) if is_var_object else 0
    return struct.pack("<QQQ", refcnt, type_id, size)
