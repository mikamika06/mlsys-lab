def check(workdir):
    from audit.engine import AuditEngine

    m = {"redundant_found": 0.0}
    engine = AuditEngine({})
    engine.enable_log()
    val = engine.find_redundant()
    m["redundant_found"] = float(val)
    return m
