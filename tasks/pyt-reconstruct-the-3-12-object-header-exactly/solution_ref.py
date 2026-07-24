import ctypes
import struct
import sys


def pack_object_header(obj):
    refcnt = sys.getrefcount(obj) - 1
    ptr = id(type(obj))
    return struct.pack("@q", refcnt) + struct.pack("@Q", ptr)
