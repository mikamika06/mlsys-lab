import ref


def check(workdir):
    from routing.router import Router
    m = {"trace_matched": 0.0}
    trace = ref.generate_trace()
    router = Router(3)
    hits = 0
    for prompt in trace:
        rep = router.step(prompt)
        matched = len(router.replica_states[rep].intersection(prompt))
        if matched > 0:
            hits += 1
        router.replica_states[rep].update(prompt)
    if hits > 0:
        m["trace_matched"] = 1.0
    return m
