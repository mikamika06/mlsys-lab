import ref


def check(workdir):
    from dsengine.scaler import simulate_trajectory

    out = {"trajectories_matched": 0.0, "overflow_handling_correct": 0.0}
    ok_count = 0
    total = len(ref.SCALER_CASES)

    for case in ref.SCALER_CASES:
        params = case["params"]
        seq = case["seq"]
        want_traj = ref.simulate_trajectory(**params, overflow_sequence=seq)
        got_traj = simulate_trajectory(**params, overflow_sequence=seq)

        if got_traj == want_traj:
            ok_count += 1
        elif "_note" not in out:
            out["_note"] = f"scaler trajectory mismatch: got {got_traj[:5]}... want {want_traj[:5]}..."

    if ok_count == total:
        out["trajectories_matched"] = 1.0
        out["overflow_handling_correct"] = 1.0

    return out
