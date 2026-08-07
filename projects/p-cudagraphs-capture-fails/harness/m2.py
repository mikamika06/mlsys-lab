def check(workdir):
    from model.net import count_dynamic_allocations
    m = {"dynamic_allocs_count": 1.0}
    try:
        count = count_dynamic_allocations()
        if count == 0:
            m["dynamic_allocs_count"] = 0.0
    except Exception:
        pass
    return m
