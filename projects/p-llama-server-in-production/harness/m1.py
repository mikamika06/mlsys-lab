def check(workdir):
    from server.diagnostics import classify_log
    m = {"classification_ok": 0.0}
    try:
        t1 = "Fatal error: Out of Memory during tensor allocation"
        t2 = "Segmentation fault (core dumped)"
        if classify_log(t1) == "oom" and classify_log(t2) == "segfault":
            m["classification_ok"] = 1.0
    except Exception:
        pass
    return m
