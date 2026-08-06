import ref


def check(workdir):
    from pipesched.schedule import identify_schedule

    out = {"schedules_matched": 0.0}
    ok = 0
    for sched in ref.SCHEDULE_TYPES:
        for seed in range(3):
            actions = ref.generate_mock_actions(sched, seed=42 + seed)
            want = ref.identify_schedule(actions)
            try:
                got = identify_schedule(actions)
            except Exception:
                got = None
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"schedule {sched} (seed {seed}): got {got}, reference {want}"
    out["schedules_matched"] = float(ok)
    return out
