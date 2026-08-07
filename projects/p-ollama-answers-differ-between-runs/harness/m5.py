def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from runner.client import ChatClient
    m = {"ten_runs_hash": 0.0}
    client = ChatClient()
    if ref.run_multiple_hashes(client, "hello", runs=10):
        m["ten_runs_hash"] = 1.0
    return m
