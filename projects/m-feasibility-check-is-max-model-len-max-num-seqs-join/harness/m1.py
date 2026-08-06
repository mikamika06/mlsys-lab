import ref

def check(workdir):
    from feasibility.check import check_feasibility
    out = {"feasibility_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.is_feasible(
            cfg["max_model_len"], cfg["max_num_seqs"], cfg["gpu_memory_bytes"],
            cfg["num_layers"], cfg["kv_heads"], cfg["head_dim"], cfg["block_size"],
            cfg["dtype_bytes"], cfg["overhead_bytes"]
        )
        got = check_feasibility(
            cfg["max_model_len"], cfg["max_num_seqs"], cfg["gpu_memory_bytes"],
            cfg["num_layers"], cfg["kv_heads"], cfg["head_dim"], cfg["block_size"],
            cfg["dtype_bytes"], cfg["overhead_bytes"]
        )
        if bool(got) == bool(want):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["feasibility_matched"] = float(ok)
    return out
