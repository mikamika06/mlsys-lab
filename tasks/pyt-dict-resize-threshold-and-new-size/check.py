import ctypes


def _dict_table_size(d):
    addr = id(d)
    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    keys_ptr = ctypes.c_void_p.from_address(addr + 4 * ptr_size).value
    if not keys_ptr:
        return 0
    log2_size = ctypes.c_uint8.from_address(keys_ptr + 8).value
    return 1 << log2_size


def _oracle(keys):
    d = {}
    sizes = []
    previous = _dict_table_size(d)
    for key in keys:
        d[key] = None
        current = _dict_table_size(d)
        if current != previous:
            sizes.append(current)
            previous = current
    return sizes


def grade(sol, fx) -> dict:
    cases = [
        list(range(0)),
        list(range(1)),
        list(range(20)),
        list(range(100)),
        [7, 3, 11, 19, 23, 31, 47, 59, 71, 83],
        [1000, -3, 42, 42, 8, 16, 24, 32, 40, 48],
    ]
    ok = 1.0
    for keys in cases:
        try:
            got = list(sol.dict_resize_sizes(list(keys)))
            ref = _oracle(list(keys))
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
