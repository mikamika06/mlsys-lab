import ref


def check(workdir):
    from ollama_evict.tracker import track_model

    out = {"states_matched": 0.0}
    ok = 0
    for scenario in ref.SCENARIOS:
        state = {}
        for name, mem, ts in scenario["ops"]:
            track_model(state, name, mem, ts)

        oracle_state = {}
        for name, mem, ts in scenario["ops"]:
            if name in oracle_state:
                oracle_state[name]["last_used"] = ts
                oracle_state[name]["access_count"] += 1
            else:
                oracle_state[name] = {"memory_bytes": mem, "last_used": ts, "access_count": 1, "loaded": True}

        match = True
        for k in oracle_state:
            if k not in state:
                match = False
                break
            if state[k]["access_count"] != oracle_state[k]["access_count"]:
                match = False
                break
            if state[k]["last_used"] != oracle_state[k]["last_used"]:
                match = False
                break
        if match:
            ok += 1
    out["states_matched"] = float(ok)
    return out
