import ref


def check(workdir):
    from runner_map.ttl import simulate_ttl
    out = {"ttl_match": 0.0}
    ok = 0
    total = len(ref.TTL_SCENARIOS)
    for scenario in ref.TTL_SCENARIOS:
        got = simulate_ttl(scenario["ttl"], scenario["ticks"])
        if got == scenario["expected_states"]:
            ok += 1
    if ok == total:
        out["ttl_match"] = 1.0
    else:
        out["_note"] = f"TTL simulation matched {ok}/{total} scenarios."
    return out
