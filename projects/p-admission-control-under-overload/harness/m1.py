def check(workdir):
    from admit.queue import OverloadQueue
    m = {"queue_ok": 0.0}
    q = OverloadQueue(2)
    if not q.push("x", 1):
        return m
    if not q.push("y", 1):
        return m
    if q.push("z", 1):
        return m
    if q.size() != 2:
        return m
    m["queue_ok"] = 1.0
    return m
