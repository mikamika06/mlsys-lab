import ref


def check(workdir):
    from gradscaler.sim import simulate_trajectory

    out = {"trajectory_matched": 0.0, "total": float(len(ref.FIXTURES))}
    ok = 0

    for i, seq in enumerate(ref.FIXTURES):
        want = ref.simulate_trajectory(seq)
        try:
            got = simulate_trajectory(seq)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Crash on sequence {i}: {e}"
            continue

        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"sequence {i}: want {want[:5]}..., got {got[:5] if isinstance(got, list) else got}"

    out["trajectory_matched"] = float(ok)
    return out
