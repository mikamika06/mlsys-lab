def check(workdir):
    from audit.engine import AuditEngine

    m = {"reorder_ratio": 1.0}
    engine = AuditEngine({})
    engine.enable_log()
    engine.optimize_sequence()
    ratio = engine.run_inference()
    m["reorder_ratio"] = float(ratio)
    return m
