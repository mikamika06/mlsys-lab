import ref


def check(workdir):
    from kquant.amortization import calculate_amortization_advantage, quantize_superblock

    out = {"error_rel_err": 1.0, "advantage_ratio_matched": 0}
    max_err = 0.0
    matched_ratios = 0

    for i, data in enumerate(ref.DATASETS):
        cfg = ref.CONFIGS[i]
        sb_size = cfg["superblock_size"]
        sub_size = cfg["subblock_size"]
        qbits = cfg["quant_bits"]

        want_recon, want_mse = ref.quantize_superblock(data, sb_size, sub_size, qbits)
        got_recon, got_mse = quantize_superblock(data, sb_size, sub_size, qbits)

        err = abs(got_mse - want_mse) / max(abs(want_mse), 1e-9)
        if err > max_err:
            max_err = err

        want_adv = ref.calculate_amortization_advantage(data, sb_size, sub_size, qbits)
        got_adv = calculate_amortization_advantage(data, sb_size, sub_size, qbits)

        adv_err = abs(got_adv["advantage_ratio"] - want_adv["advantage_ratio"]) / max(abs(want_adv["advantage_ratio"]), 1e-9)
        if adv_err <= 1e-5:
            matched_ratios += 1

    out["error_rel_err"] = float(max_err)
    out["advantage_ratio_matched"] = 1 if matched_ratios == len(ref.DATASETS) else 0
    return out
