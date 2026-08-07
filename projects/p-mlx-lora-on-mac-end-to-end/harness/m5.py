def check(workdir):
    import ref
    from lora_pipe import engine
    m = {"server_ok": 0.0}
    try:
        server = engine.LoraServer("dummy")
        resp = server.handle_request("ping")
        if isinstance(resp, str) and len(resp) > 0:
            m["server_ok"] = 1.0
    except Exception:
        pass
    return m
