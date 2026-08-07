import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from numa_tuning import affinity

    m = {"pinned_correctly": 0.0, "numa_local": 0.0}
    try:
        res = affinity.apply_pinning(0, 4)
        if isinstance(res, dict) and res.get("status") == "success":
            m["pinned_correctly"] = 1.0

        mem = affinity.allocate_numa_memory(1024, 0)
        if isinstance(mem, dict) and mem.get("allocated") is True:
            m["numa_local"] = 1.0
    except Exception:
        pass
    return m
