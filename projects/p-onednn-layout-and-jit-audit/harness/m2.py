def check(workdir):
    from audit.engine import AuditEngine

    m = {"transitions_detected": 0.0}
    engine = AuditEngine({})
    engine.enable_log()
    trans = engine.get_transitions()
    m["transitions_detected"] = float(len(trans))
    return m
