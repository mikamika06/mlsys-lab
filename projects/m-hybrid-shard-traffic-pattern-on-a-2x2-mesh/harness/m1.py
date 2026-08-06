import ref


def check(workdir):
    from mesh.traffic import simulate_traffic

    out = {"traffic_matched": 0.0}
    ok = 0
    for strat in ref.STRATEGIES:
        want = ref.simulate_traffic(strat, (2, 2))
        try:
            got = simulate_traffic(strat, (2, 2))
        except Exception:
            got = {}
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"strategy {strat}: got {got}, reference {want}"
    out["traffic_matched"] = float(ok)
    return out
