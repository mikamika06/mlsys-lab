import ctypes


def _dict_table_size(d):
    addr = id(d)
    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    keys_ptr = ctypes.c_void_p.from_address(addr + 4 * ptr_size).value
    if not keys_ptr:
        return 0
    log2_size = ctypes.c_uint8.from_address(keys_ptr + 8).value
    return 1 << log2_size


def dict_resize_sizes(keys):
    d = {}
    result = []
    previous = _dict_table_size(d)
    for key in keys:
        d[key] = None
        current = _dict_table_size(d)
        if current != previous:
            result.append(current)
            previous = current
    return result
