import ref


def check(workdir):
    from converter.fallback import analyze_fallback_vs_rewrite

    out = {"latency_evaluated": 0.0, "fallback_decisions_match": 0.0}

    nodes = ref.TEST_GRAPHS[0]["nodes"]
    metrics = ref.TEST_METRICS

    got = analyze_fallback_vs_rewrite(nodes, metrics)

    exp_decisions = {}
    exp_total = 0.0
    for node in nodes:
        nid = node["id"]
        op = node["op_type"]
        m = metrics.get(op, {})
        fb = m.get("custom_op_overhead", 10.0) + m.get("fallback_exec", 5.0)
        rw = m.get("decomposition_exec", 8.0)
        if rw <= fb:
            strat = "REWRITE"
            lat = rw
        else:
            strat = "FALLBACK"
            lat = fb
        exp_decisions[nid] = {"strategy": strat, "latency": lat}
        exp_total += lat

    if isinstance(got, dict) and "decisions" in got and "total_latency" in got:
        out["latency_evaluated"] = 1.0
        if (got["decisions"] == exp_decisions and
            abs(got["total_latency"] - exp_total) < 1e-5):
            out["fallback_decisions_match"] = 1.0
        else:
            out["_note"] = f"Expected decisions {exp_decisions}, got {got.get('decisions')}"
    else:
        out["_note"] = "Return dictionary missing required keys 'decisions' or 'total_latency'"

    return out
