import ref


def check(workdir):
    from auditor.layout import audit_reused_blobs

    out = {"layout_matched": 0.0}
    ok = 0
    for case in ref.REVISIONS:
        want = ref.audit_reused_blobs(case["rev1"], case["rev2"])
        got = audit_reused_blobs(case["rev1"], case["rev2"])
        if got == want:
            ok += 1
    out["layout_matched"] = float(ok)
    return out
