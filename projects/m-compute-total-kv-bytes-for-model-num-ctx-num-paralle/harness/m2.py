import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import slots.runner as runner
    except ImportError:
        return {"completions_match": 0.0, "drops_match": 0.0, "metrics_match": 0.0}

    import ref

    reqs1 = [
        {"id": "r1", "arrival": 0, "tokens": 10},
        {"id": "r2", "arrival": 0, "tokens": 20},
        {"id": "r3", "arrival": 0, "tokens": 10},
        {"id": "r4", "arrival": 5, "tokens": 10},
        {"id": "r5", "arrival": 5, "tokens": 5}
    ]
    reqs2 = [{"id": f"r{i}", "arrival": i*5, "tokens": 10} for i in range(10)]

    cases = [
        (reqs1, 2, 1, 10),
        (reqs1, 1, 1, 10),
        (reqs2, 1, 2, 10),
        (reqs2, 4, 10, 5)
    ]

    comp, drop, met = 0, 0, 0

    for c in cases:
        try:
            r_want = ref.simulate(*c)
            r_got = runner.simulate(*c)

            if r_want["completed"] == r_got.get("completed"):
                comp += 1
            if r_want["dropped"] == r_got.get("dropped"):
                drop += 1

            agg_diff = abs(r_want["aggregate_tok_s"] - r_got.get("aggregate_tok_s", -1))
            if agg_diff < 0.1 and r_want["latencies"] == r_got.get("latencies"):
                met += 1
        except Exception:
            pass

    return {
        "completions_match": float(comp) / len(cases),
        "drops_match": float(drop) / len(cases),
        "metrics_match": float(met) / len(cases)
    }
