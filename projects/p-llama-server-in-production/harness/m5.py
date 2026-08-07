def check(workdir):
    from server.engine import simulate_load
    m = {"endurance_ok": 0.0}
    res = simulate_load(1, 10)
    if res["failures"] == 0 and res["uptime_pct"] == 100.0:
        m["endurance_ok"] = 1.0
    return m
