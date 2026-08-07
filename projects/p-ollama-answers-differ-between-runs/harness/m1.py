def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from runner.client import ChatClient
    m = {"divergence_measured": 0.0}
    client = ChatClient()
    div = ref.measure_divergence(client, "test prompt", runs=5)
    if 0.0 <= div <= 1.0:
        m["divergence_measured"] = 1.0
    return m
