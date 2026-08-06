import ref

def check(workdir):
    from fsdp_shard.padding import compute_padding_overhead
    out = {"padding_matched": 0.0, "_note": ""}
    ok = 0
    tests = [
        ([103, 200], 4),
        ([1025, 2047], 8),
        ([512, 512], 3),
    ]
    for i, (params, ws) in enumerate(tests):
        want = ref.compute_padding_overhead(params, ws)
        got = compute_padding_overhead(params, ws)
        if abs(got - want) < 1e-5:
            ok += 1
        elif not out["_note"]:
            out["_note"] = f"test {i}: got {got}, want {want}"
    out["padding_matched"] = 1.0 if ok == len(tests) else 0.0
    return out
