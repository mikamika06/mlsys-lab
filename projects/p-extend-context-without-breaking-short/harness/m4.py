def check(workdir):
    from ctx.scaling import verify_retrieval
    m = {"retrieval_verified": 0.0}
    try:
        model = lambda c, q: {"found": 0.99}
        res = verify_retrieval(model, "context", "query")
        if res > 0.9:
            m["retrieval_verified"] = 1.0
    except Exception:
        pass
    return m
