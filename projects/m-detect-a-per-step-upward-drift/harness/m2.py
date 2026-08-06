import ref

def check(workdir):
    from leak.snapshots import classify_snapshots
    cases = ref.get_snapshot_cases()
    for case in cases:
        got = classify_snapshots(case)
        want = ["leaking", "stable", "cached", "leaking", "stable", "cached"]
        if got != want:
            return {"classification_matched": 0.0}
    return {"classification_matched": 1.0}
