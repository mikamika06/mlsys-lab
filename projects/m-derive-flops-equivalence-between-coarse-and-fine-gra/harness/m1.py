import ref


def check(workdir):
    from moe.flops import derive_fine_grained_split
    from moe.combinatorics import count_reachable_combinations

    out = {"flops_match": 0.0, "splits_match": 0.0}
    flops_ok = 0
    splits_ok = 0

    for cfg in ref.CONFIGS:
        want_split = ref.derive_fine_grained_split(
            cfg["d_model"], cfg["d_ffn_coarse"], cfg["num_coarse"],
            cfg["k_coarse"], cfg["num_shared"], cfg["k_fine"], cfg["split_factor"]
        )
        got_split = derive_fine_grained_split(
            cfg["d_model"], cfg["d_ffn_coarse"], cfg["num_coarse"],
            cfg["k_coarse"], cfg["num_shared"], cfg["k_fine"], cfg["split_factor"]
        )

        if got_split == want_split:
            flops_ok += 1

        want_coarse_comb = ref.count_reachable_combinations(cfg["num_coarse"], cfg["k_coarse"])
        want_fine_comb = ref.count_reachable_combinations(cfg["num_routed_fine"], cfg["k_fine"])

        got_coarse_comb = count_reachable_combinations(cfg["num_coarse"], cfg["k_coarse"])
        got_fine_comb = count_reachable_combinations(cfg["num_routed_fine"], cfg["k_fine"])

        if got_coarse_comb == want_coarse_comb and got_fine_comb == want_fine_comb:
            splits_ok += 1

    out["flops_match"] = 1.0 if flops_ok == len(ref.CONFIGS) else 0.0
    out["splits_match"] = 1.0 if splits_ok == len(ref.CONFIGS) else 0.0
    return out
