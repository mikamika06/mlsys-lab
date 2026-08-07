def check(workdir):
    import ref
    from kv.allocator import PagedKVAllocator

    m = {"below_threshold": 0.0, "memory_saved": 0.0}
    try:
        trace = ref.get_beam_search_trace()
        a = PagedKVAllocator(100, 4)

        max_used = 0
        for op in trace:
            if op[0] == "alloc":
                a.allocate_seq(op[1])
            elif op[0] == "append":
                a.append_token(op[1])
            elif op[0] == "fork":
                a.fork_seq(op[1], op[2])
            elif op[0] == "free":
                a.free_seq(op[1])

            used = 100 - a.free_count()
            if used > max_used:
                max_used = used

        if max_used <= 50:
            m["below_threshold"] = 1.0
            m["memory_saved"] = 1.0
    except Exception:
        pass
    return m
