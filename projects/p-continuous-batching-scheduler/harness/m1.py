def check(workdir):
    from sched import Allocator

    m = {"api_ok": 0.0, "freelist_order": 0.0, "refcount_ok": 0.0,
         "double_free_safe": 0.0, "keys_ok": 0.0, "exhaustion_ok": 0.0}
    a = Allocator(4, 16)
    if a.free_count() != 4:
        return m
    m["api_ok"] = 1.0

    b0, b1, b2 = a.allocate(), a.allocate(), a.allocate()
    if (b0, b1, b2) != (0, 1, 2) or a.free_count() != 1:
        return m
    a.release(b1)
    if a.free_count() != 2 or a.allocate() != 1:
        return m
    m["freelist_order"] = 1.0

    a2 = Allocator(2, 16)
    x = a2.allocate()
    a2.share(x)
    a2.release(x)
    if a2.free_count() != 1:
        return m
    a2.release(x)
    if a2.free_count() != 2:
        return m
    m["refcount_ok"] = 1.0

    a2.release(x)
    m["double_free_safe"] = 1.0 if a2.free_count() == 2 else 0.0

    a3 = Allocator(4, 16)
    y = a3.allocate()
    a3.register(y, "k")
    if a3.lookup("k") != y or a3.lookup("missing") is not None:
        return m
    a3.release(y)
    m["keys_ok"] = 1.0 if a3.lookup("k") is None else 0.0

    a4 = Allocator(1, 16)
    a4.allocate()
    try:
        a4.allocate()
        m["exhaustion_ok"] = 0.0
    except Exception:
        m["exhaustion_ok"] = 1.0
    return m
