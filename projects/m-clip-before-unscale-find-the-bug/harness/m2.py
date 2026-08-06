import ref


def check(workdir):
    from scalerlab.counter import UnderflowTracker

    tracker = UnderflowTracker()
    statuses = [True, False, True, True, False, False, True]
    for s in statuses:
        tracker.update(s)

    skipped = tracker.get_skipped_count()
    expected_skipped = sum(1 for s in statuses if s)

    out = {
        "skipped_match": 1.0 if skipped == expected_skipped else 0.0,
        "behavior_match": 1.0 if tracker.total_steps == len(statuses) else 0.0
    }
    return out
