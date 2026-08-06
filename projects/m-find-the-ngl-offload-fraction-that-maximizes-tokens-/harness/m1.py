import ref


def check(workdir):
    from metalopt.offload import find_optimal_ngl, calculate_throughput_ratio

    optimal_matched = 0
    ratio_valid = 0
    total = len(ref.CONFIGS)

    for i, cfg in enumerate(ref.CONFIGS):
        layers = cfg["layers"]
        vram = cfg["vram_limit_mb"]
        mem = cfg["layer_memory_mb"]

        want_ngl = ref.compute_reference_ngl(cfg)
        got_ngl, got_tps = find_optimal_ngl(layers, vram, mem)

        if got_ngl == want_ngl:
            optimal_matched += 1

        frac = got_ngl / float(layers) if layers > 0 else 0.0
        ratio = calculate_throughput_ratio(frac, 2.0, 15.0, got_ngl * mem, vram)
        if ratio > 0.0:
            ratio_valid += 1

    out = {
        "optimal_ngl_matched": 1.0 if optimal_matched == total else 0.0,
        "throughput_ratio_valid": 1.0 if ratio_valid == total else 0.0,
    }
    if optimal_matched != total:
        out["_note"] = f"matched {optimal_matched}/{total} optimal ngl configs"
    return out
