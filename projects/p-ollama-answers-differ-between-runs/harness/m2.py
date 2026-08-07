def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from runner.config import merge_options
    m = {"priority_correct": 0.0}
    mf = {"temperature": 0.7, "seed": 1}
    api = {"temperature": 0.5}
    req = {"temperature": 0.0}
    res = merge_options(mf, api, req)
    if res.get("temperature") == 0.0 and res.get("seed") == 1:
        m["priority_correct"] = 1.0
    return m
