import ref


def check(workdir):
    from routing.router import Router
    m = {"high_hit_rate_no_skew": 0.0}
    trace = ref.generate_trace() * 3
    router = Router(2)
    hits = 0
    for prompt in trace:
        rep = router.step(prompt)
        matched = len(router.replica_states[rep].intersection(prompt))
        if matched > 0:
            hits += 1
        router.replica_states[rep].update(prompt)
    hit_rate = hits / len(trace)
    if hit_rate > 0.2:
        m["high_hit_rate_no_skew"] = 1.0
    return m
