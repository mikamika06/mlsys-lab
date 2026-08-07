def check(workdir):
    m = {"surge_handled": 0.0}
    try:
        from batching.controller import DynamicBatchController
    except Exception:
        return m

    try:
        ctrl = DynamicBatchController({1: 10.0, 8: 30.0}, 50.0)
        incoming = [{"id": i, "size": 1} for i in range(20)]
        batch = ctrl.step(incoming, 0.9)
        if isinstance(batch, list) and len(batch) > 0:
            m["surge_handled"] = 1.0
    except Exception:
        return m
    return m
