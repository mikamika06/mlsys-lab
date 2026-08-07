def check(workdir):
    from moe.cache import ExpertCache
    from moe.policy import evaluate_latency
    import ref

    m = {"latency_under_threshold": 0.0}
    cache = ExpertCache(5000)
    traces = ref.get_sample_traces()
    lat = evaluate_latency(traces, cache, 5000)
    if lat < 20.0:
        m["latency_under_threshold"] = 1.0
    return m
