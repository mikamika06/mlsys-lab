def check(workdir):
    import ref
    from kv.allocator import PagedKVAllocator

    m = {"no_crash": 0.0, "no_leaks": 0.0}
    try:
        trace = ref.get_random_trace(10000)
        a = PagedKVAllocator(2000, 4)

        for op in trace:
            if op[0] == "alloc":
                a.allocate_seq(op[1])
            elif op[0] == "append":
                try:
                    a.append_token(op[1])
                except RuntimeError:
                    pass
            elif op[0] == "fork":
                a.fork_seq(op[1], op[2])
            elif op[0] == "free":
                a.free_seq(op[1])

        m["no_crash"] = 1.0
        if a.free_count() == 2000:
            m["no_leaks"] = 1.0
    except Exception:
        pass
    return m
