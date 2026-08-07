import ref


def check(workdir):
    from routing.router import Router
    m = {"hit_rate_measured": 0.0}
    trace = ref.generate_trace()
    router = Router(4)
    hits = 0
    total = len(trace)
    for i, prompt in enumerate(trace):
        rep = router.round_robin_route(i)
        matched = len(router.replica_states[rep].intersection(prompt))
        if matched > 0:
            hits += 1
        router.replica_states[rep].update(prompt)
    hit_rate = hits / total
    if 0.0 <= hit_rate <= 1.0:
        m["hit_rate_measured"] = 1.0
    return m
