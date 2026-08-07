import ref


def check(workdir):
    from swapspace.allocator import simulate_preemption_trajectory

    out = {
        "trajectory_match": 0.0,
        "peak_bytes_match": 0.0,
    }

    cfg = ref.CONFIGS[0]
    events = ref.PREEMPTION_EVENTS

    want = ref.simulate_preemption_trajectory(cfg, events)
    got = simulate_preemption_trajectory(cfg, events)

    if got.get("trajectory") == want["trajectory"]:
        out["trajectory_match"] = 1.0
    else:
        out["_note"] = (
            f"trajectory mismatch: got {got.get('trajectory')}, want {want['trajectory']}"
        )

    if got.get("peak_bytes") == want["peak_bytes"]:
        out["peak_bytes_match"] = 1.0
    elif "_note" not in out:
        out["_note"] = (
            f"peak_bytes mismatch: got {got.get('peak_bytes')}, want {want['peak_bytes']}"
        )

    return out
