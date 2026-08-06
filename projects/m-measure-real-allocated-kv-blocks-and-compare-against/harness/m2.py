import ref

def check(workdir):
    from kvalloc.simulator import measure_allocated_blocks
    from kvalloc.metrics import compute_relative_error
    out = {"rel_err_match": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want_blocks = ref.measure_allocated_blocks(cfg["block_tables"], cfg["block_size"])
        got_blocks = measure_allocated_blocks(cfg["block_tables"], cfg["block_size"])
        want_err = ref.compute_relative_error(got_blocks, want_blocks)
        got_err = compute_relative_error(got_blocks, want_blocks)
        if got_blocks == want_blocks and abs(got_err - want_err) < 1e-5:
            ok += 1
    if ok == len(ref.CONFIGS):
        out["rel_err_match"] = 1.0
    return out
