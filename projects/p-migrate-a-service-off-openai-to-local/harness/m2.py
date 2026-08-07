def check(workdir):
    from runner import adapter

    m = {"failures_collected": 0.0}
    try:
        fails = adapter.collect_failures()
        if isinstance(fails, list) and len(fails) > 0:
            m["failures_collected"] = 1.0
    except Exception:
        pass
    return m
