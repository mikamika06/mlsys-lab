def check(workdir):
    from model.net import locate_breaking_operation
    m = {"faulty_ops_found": 0.0}
    try:
        ops = locate_breaking_operation()
        if isinstance(ops, list) and len(ops) > 0:
            m["faulty_ops_found"] = 1.0
    except Exception:
        pass
    return m
