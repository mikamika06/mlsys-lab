import ref

def check(workdir):
    from fsdp_shard.inspect import inspect_shard
    out = {"shards_matched": 0.0, "_note": ""}
    ok = 0
    tests = [
        ([100, 200, 300], 4, 0),
        ([1024, 2048], 2, 1),
        ([50, 50, 50, 50], 2, 0),
    ]
    for i, (params, ws, r) in enumerate(tests):
        want = ref.inspect_shard(params, ws, r)
        got = inspect_shard(params, ws, r)
        if got == want:
            ok += 1
        elif not out["_note"]:
            out["_note"] = f"test {i}: got {got}, want {want}"
    out["shards_matched"] = float(ok)
    return out
