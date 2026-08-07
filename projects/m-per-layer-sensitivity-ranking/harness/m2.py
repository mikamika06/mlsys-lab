import ref


def check(workdir):
    from quant.allocation import allocate_bits
    from quant.groups import emit_config_groups
    from quant.sensitivity import compute_sensitivities

    out = {"allocation_match": 0.0}
    ok = 0
    total = len(ref.CONFIGS)
    bit_options = [2, 4, 8]
    for cfg in ref.CONFIGS:
        sens = compute_sensitivities(cfg, ref.STATS)
        budget = len(cfg["layers"]) * 4
        want_bits = ref.allocate_bits(sens, bit_options, budget)
        got_bits = allocate_bits(sens, bit_options, budget)
        want_groups = ref.emit_config_groups(cfg, want_bits)
        got_groups = emit_config_groups(cfg, got_bits)
        if got_bits == want_bits and got_groups == want_groups:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got bits {got_bits}, reference {want_bits}"
    if ok == total:
        out["allocation_match"] = 1.0
    return out
