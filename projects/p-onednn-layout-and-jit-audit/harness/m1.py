def check(workdir):
    from audit.engine import AuditEngine

    m = {"log_active": 0.0, "events_parsed": 0.0}
    engine = AuditEngine({})
    res = engine.enable_log()
    if res != 1:
        return m
    m["log_active"] = 1.0

    events = engine.parse_events()
    m["events_parsed"] = float(len(events))
    return m
