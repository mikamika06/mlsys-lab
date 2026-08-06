import ref

def check(workdir):
    from router.affinity import select_replica
    replicas = [
        {"load": 2, "cache": {1, 2}},
        {"load": 8, "cache": {1}},
        {"load": 2, "cache": {3}}
    ]
    got = select_replica(replicas, 1, max_load_diff=3)
    want = ref.select_replica(replicas, 1, max_load_diff=3)
    out = {"affinity_match": 1.0 if got == want else 0.0}
    if got != want:
        out["_note"] = f"got replica {got}, want {want}"
    return out
