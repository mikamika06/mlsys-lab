import ref

def check(workdir):
    from partitioner.predict import partition_ops
    
    out = {"partitions_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.partition_ops(cfg['ops'], cfg['base_allowlist'])
        got = partition_ops(cfg['ops'], cfg['base_allowlist'])
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
            
    out["partitions_matched"] = float(ok)
    return out
