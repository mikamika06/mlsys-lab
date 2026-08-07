import ref


def check(workdir):
    from scaler.trajectory import simulate_trajectory

    out = {"trajectory_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        history = cfg["history"]
        initial = cfg["initial"]
        interval = cfg.get("growth_interval", cfg.get("interval", 2000))
        want = ref.simulate_trajectory(history, initial_scale=initial, growth_interval=interval)
        try:
            got = simulate_trajectory(history, initial_scale=initial, growth_interval=interval)
        except Exception:
            got = []
        if len(got) == len(want) and all(abs(a - b) < 1e-5 for a, b in zip(got, want)):
            ok += 1
        else:
            out["_note"] = f"got {got}, want {want}"
            break
    if ok == len(ref.CONFIGS):
        out["trajectory_matched"] = 1.0
    return out
