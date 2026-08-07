import ref

def check(workdir):
    from edge.partition import partition_graph
    out = {"ratio_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.partition_graph(cfg, cfg)
        got = partition_graph(cfg, cfg)
        if got.get("delegated") == want["delegated"] and got.get("host") == want["host"]:
            ok += 1
    if ok == len(ref.CONFIGS):
        out["ratio_matched"] = 1.0
    return out
