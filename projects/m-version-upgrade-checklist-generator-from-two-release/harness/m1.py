import ref


def check(workdir):
    from upgradeprep.parser import parse_release_notes

    out = {"snapshots_matched": 0.0}
    ok = 0
    for i, raw in enumerate(ref.RAW_RELEASE_NOTES):
        want = ref.parse_release_notes(raw)
        got = parse_release_notes(raw)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"snapshot {i}: got {got}, expected {want}"
    out["snapshots_matched"] = float(ok)
    return out
