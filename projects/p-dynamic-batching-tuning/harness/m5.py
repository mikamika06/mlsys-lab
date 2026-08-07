def check(workdir):
    m = {"slo_satisfied": 0.0}
    try:
        from batching.controller import DynamicBatchController
    except Exception:
        return m

    try:
        ctrl = DynamicBatchController({1: 10.0, 8: 25.0}, 30.0)
        batch = ctrl.step([{"id": 1, "size": 1}], 0.1)
        if isinstance(batch, list):
            m["slo_satisfied"] = 1.0
    except Exception:
        return m
    return m
