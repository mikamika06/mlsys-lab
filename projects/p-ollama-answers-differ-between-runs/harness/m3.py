def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from runner.client import ChatClient
    m = {"params_forwarded": 0.0}
    client = ChatClient()
    payload = client.prepare_payload("hello", {"seed": 42, "temperature": 0.0})
    if "options" in payload and payload["options"].get("seed") == 42:
        m["params_forwarded"] = 1.0
    return m
