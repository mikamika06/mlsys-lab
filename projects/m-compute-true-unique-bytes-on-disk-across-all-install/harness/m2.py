import ref


def check(workdir):
    from blobstore import (
        unique_bytes_on_disk,
        naive_total_bytes,
        incremental_pull_bytes,
        find_orphaned_blobs,
        orphaned_bytes,
    )

    out = {"unique_match": 0.0, "naive_match": 0.0,
           "incremental_match": 0.0, "orphan_match": 0.0}

    unique_ok = naive_ok = incr_ok = orphan_ok = 0
    notes = []
    for i, case in enumerate(ref.CASES):
        cfg = case["config"]
        candidate = case["candidate"]
        disk = case["disk_blobs"]

        want_unique = ref.unique_bytes_on_disk(cfg)
        try:
            got_unique = unique_bytes_on_disk(cfg)
        except Exception:
            got_unique = None
        if got_unique == want_unique:
            unique_ok += 1
        elif len(notes) < 4:
            notes.append(f"case {i}: unique_bytes_on_disk got {got_unique}, reference {want_unique}")

        want_naive = ref.naive_total_bytes(cfg)
        try:
            got_naive = naive_total_bytes(cfg)
        except Exception:
            got_naive = None
        if got_naive == want_naive:
            naive_ok += 1
        elif len(notes) < 4:
            notes.append(f"case {i}: naive_total_bytes got {got_naive}, reference {want_naive}")

        want_incr = ref.incremental_pull_bytes(cfg, candidate)
        try:
            got_incr = incremental_pull_bytes(cfg, candidate)
        except Exception:
            got_incr = None
        if got_incr == want_incr:
            incr_ok += 1
        elif len(notes) < 4:
            notes.append(f"case {i}: incremental_pull_bytes got {got_incr}, reference {want_incr}")

        want_orphans = ref.find_orphaned_blobs(cfg, disk)
        want_orphan_bytes = ref.orphaned_bytes(cfg, disk)
        try:
            got_orphans = sorted(find_orphaned_blobs(cfg, disk) or [])
            got_orphan_bytes = orphaned_bytes(cfg, disk)
        except Exception:
            got_orphans, got_orphan_bytes = None, None
        if got_orphans == want_orphans and got_orphan_bytes == want_orphan_bytes:
            orphan_ok += 1
        elif len(notes) < 4:
            notes.append(
                f"case {i}: orphans got {got_orphans}/{got_orphan_bytes}, "
                f"reference {want_orphans}/{want_orphan_bytes}"
            )

    n = len(ref.CASES)
    out["unique_match"] = 1.0 if unique_ok == n else 0.0
    out["naive_match"] = 1.0 if naive_ok == n else 0.0
    out["incremental_match"] = 1.0 if incr_ok == n else 0.0
    out["orphan_match"] = 1.0 if orphan_ok == n else 0.0
    if notes:
        out["_note"] = " | ".join(notes)
    return out
