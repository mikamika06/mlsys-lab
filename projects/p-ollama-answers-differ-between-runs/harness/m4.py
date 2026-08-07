def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from runner.client import ChatClient
    m = {"byte_for_byte": 0.0}
    client = ChatClient()
    out1 = client.generate("hello", seed=42, temperature=0.0, num_predict=32)
    out2 = client.generate("hello", seed=42, temperature=0.0, num_predict=32)
    if out1 == out2 and len(out1) > 0:
        m["byte_for_byte"] = 1.0
    return m
