import gc


def _scenario_full(sol, n=5):
    events = []
    g = sol.make_managed_gen(events, n)
    got = list(g)
    del g
    gc.collect()
    return got == list(range(n)) and events == ["acquire", "release"]


def _scenario_close(sol, n=5, k=2):
    events = []
    g = sol.make_managed_gen(events, n)
    got = [next(g) for _ in range(k)]
    if events != ["acquire"]:
        return False
    g.close()
    del g
    gc.collect()
    return got == list(range(k)) and events == ["acquire", "release"]


def _scenario_abandon(sol, n=5, k=1):
    events = []
    g = sol.make_managed_gen(events, n)
    got = [next(g) for _ in range(k)]
    if events != ["acquire"]:
        return False
    del g
    gc.collect()
    return got == list(range(k)) and events == ["acquire", "release"]


def grade(sol, fx) -> dict:
    try:
        checks = [
            _scenario_full(sol),
            _scenario_close(sol),
            _scenario_abandon(sol),
            # a second, differently-sized run to catch hardcoding to n=5
            _scenario_full(sol, n=3),
            _scenario_close(sol, n=8, k=5),
        ]
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if all(checks) else 0.0}
