import ref


def check(workdir):
    from engine.blast import estimate_blast

    data = ref.BATCH_DATA
    got = estimate_blast(data["requests"], data["crash_index"])
    want = {
        "lost_count": 2,
        "lost_ids": ["req_102", "req_103"],
        "retried_count": 1
    }
    match = 1 if got == want else 0
    out = {"blast_match": float(match)}
    if not match:
        out["_note"] = f"got {got}, want {want}"
    return out
