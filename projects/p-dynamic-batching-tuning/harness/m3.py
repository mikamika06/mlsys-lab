def check(workdir):
    m = {"queues_separated": 0.0}
    try:
        from batching.queues import TieredQueueManager
    except Exception:
        return m

    try:
        qm = TieredQueueManager([10, 50])
        qm.push({"id": "a", "size": 5})
        qm.push({"id": "b", "size": 100})
        batch = qm.pop_batch(2)
        if len(batch) == 2 and batch[0]["id"] == "a":
            m["queues_separated"] = 1.0
    except Exception:
        return m
    return m
