def audit_reused_blobs(rev1, rev2):
    set1 = set(rev1.items())
    set2 = set(rev2.items())
    return {
        "reused": dict(set1.intersection(set2)),
        "only_rev1": dict(set1 - set2),
        "only_rev2": dict(set2 - set1)
    }
