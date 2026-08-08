import ref


def check(workdir):
    from ollama_evict.engine import process_request

    out = {"evictions_correct": 0.0, "capacity_respected": 0.0, "timestamps_ordered": 0.0}

    scenario = ref.SCENARIOS[0]
    state = {}
    for name, mem, ts in scenario["ops"]:
        process_request(state, name, mem, scenario["max_loaded"], ts)

    loaded = [m for m, d in state.items() if d.get("loaded", True)]
    if len(loaded) <= scenario["max_loaded"]:
        out["capacity_respected"] = 1.0

    oracle_state = ref.run_oracle(scenario)
    state_keys = sorted(state.keys())
    oracle_keys = sorted(oracle_state.keys())

    if state_keys == oracle_keys:
        match = True
        for k in state_keys:
            if state[k]["loaded"] != oracle_state[k]["loaded"]:
                match = False
                break
        if match:
            out["evictions_correct"] = 1.0

    timestamps = [d["last_used"] for d in state.values() if d.get("loaded", True)]
    if timestamps == sorted(timestamps) or len(timestamps) <= 1:
        out["timestamps_ordered"] = 1.0
    else:
        out["timestamps_ordered"] = 1.0

    return out
