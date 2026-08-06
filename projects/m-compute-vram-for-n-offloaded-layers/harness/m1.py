import ref


def check(workdir):
    from offload.compute import compute_vram

    out = {"vram_matched": 0.0}
    ok = 0
    total = 0
    for cfg in ref.CONFIGS:
        for ngl in range(cfg["num_layers"] + 1):
            total += 1
            if ref.compute_vram(cfg, ngl) == compute_vram(cfg, ngl):
                ok += 1
    out["vram_matched"] = ok / total if total > 0 else 0.0
    return out
