def check(workdir):
    from audit.engine import AuditEngine

    m = {"sequence_optimized": 0.0}
    engine = AuditEngine({})
    engine.enable_log()
    val = engine.optimize_sequence()
    m["sequence_optimized"] = float(val)
    return m
