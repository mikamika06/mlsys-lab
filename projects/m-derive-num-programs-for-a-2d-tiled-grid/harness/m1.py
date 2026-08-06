import ref

def check(workdir):
    from triton_grid.config import derive_num_programs

    out = {"configs_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.derive_num_programs(cfg["width"], cfg["height"], cfg["block_w"], cfg["block_h"])
        try:
            got = derive_num_programs(cfg["width"], cfg["height"], cfg["block_w"], cfg["block_h"])
            if got == want:
                ok += 1
        except Exception:
            pass
    out["configs_matched"] = float(ok)
    return out
