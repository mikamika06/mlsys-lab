import ref


def check(workdir):
    from trtcache.engine import classify_init_state

    out = {"states_matched": 0.0}
    ok = 0
    for i, sc in enumerate(ref.SCENARIOS):
        got = classify_init_state(sc["engine_meta"], sc["cache_store"])
        if got == sc["expected"]:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got {got}, want {sc['expected']}"
    out["states_matched"] = float(ok)
    return out
