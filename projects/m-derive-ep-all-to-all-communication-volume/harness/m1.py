import ref

def check(workdir):
    from epall.volume import compute_ep_all_to_all_volume
    out = {"volume_match": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_volume(cfg["world_size"], cfg["num_tokens"], cfg["hidden_size"], cfg["top_k"], cfg["dtype_size"])
        got = compute_ep_all_to_all_volume(cfg["world_size"], cfg["num_tokens"], cfg["hidden_size"], cfg["top_k"], cfg["dtype_size"])
        if abs(want - got) < 1e-5:
            ok += 1
    out["volume_match"] = 1.0 if ok == len(ref.CONFIGS) else 0.0
    return out
