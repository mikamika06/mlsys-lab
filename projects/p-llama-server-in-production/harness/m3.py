def check(workdir):
    from server.slots import configure_slots
    m = {"slots_config_ok": 0.0}
    try:
        cfg = configure_slots(4096, 4, "q4_0")
        if cfg["slot_size"] == 1024 and cfg["kv_bits"] == 4:
            m["slots_config_ok"] = 1.0
    except Exception:
        pass
    return m
