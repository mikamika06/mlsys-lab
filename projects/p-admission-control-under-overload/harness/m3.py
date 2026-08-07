def check(workdir):
    from admit.queue import OverloadQueue
    m = {"priority_ok": 0.0}
    q = OverloadQueue(5)
    q.push("low", 1)
    q.push("high", 10)
    q.push("mid", 5)
    if q.pop() != "high":
        return m
    if q.pop() != "mid":
        return m
    if q.pop() != "low":
        return m
    m["priority_ok"] = 1.0
    return m
