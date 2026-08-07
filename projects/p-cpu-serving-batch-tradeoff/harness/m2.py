def check(workdir):
    from serving import engine
    m = {"threads_ok": 0.0}
    try:
        res = engine.thread_scaling([1, 4, 8], [2, 4], 10.0)
        if 2 in res and 4 in res and len(res[2]) == 3:
            m["threads_ok"] = 1.0
    except Exception:
        pass
    return m
