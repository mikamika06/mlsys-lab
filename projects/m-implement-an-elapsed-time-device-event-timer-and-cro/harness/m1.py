import ref


def check(workdir):
    from timer.events import measure_elapsed_time

    out = {"events_matched": 0.0}
    ok = 0
    for p in ref.PROFILES:
        want = ref.simulate_events(p)
        got = measure_elapsed_time(p)
        if abs(got - want) < 0.5:
            ok += 1
    out["events_matched"] = float(ok)
    return out
