import ref


def check(workdir):
    from upgradeprep.parser import parse_release_notes
    from upgradeprep.diff import compute_upgrade_diff
    from upgradeprep.checklist import generate_checklist

    out = {"diffs_matched": 0.0, "checklists_matched": 0.0}

    snaps = [parse_release_notes(raw) for raw in ref.RAW_RELEASE_NOTES]
    ref_snaps = [ref.parse_release_notes(raw) for raw in ref.RAW_RELEASE_NOTES]

    diff_ok = True
    checklist_ok = True

    for i in range(len(snaps) - 1):
        want_diff = ref.compute_upgrade_diff(ref_snaps[i], ref_snaps[i + 1])
        got_diff = compute_upgrade_diff(snaps[i], snaps[i + 1])
        if got_diff != want_diff:
            diff_ok = False
            if "_note" not in out:
                out["_note"] = f"diff mismatch at pair {i}: got {got_diff}, expected {want_diff}"
            break

        want_cl = ref.generate_checklist(want_diff)
        got_cl = generate_checklist(got_diff)
        if got_cl != want_cl:
            checklist_ok = False
            if "_note" not in out:
                out["_note"] = f"checklist mismatch at pair {i}: got {got_cl}, expected {want_cl}"
            break

    if diff_ok:
        out["diffs_matched"] = 1.0
    if checklist_ok:
        out["checklists_matched"] = 1.0

    return out
