import math


def count_reachable_combinations(num_routed, top_k, num_shared=0):
    if top_k > num_routed:
        return 0
    return math.comb(num_routed, top_k)


def compare_combinations(coarse_cfg, fine_cfg):
    coarse_comb = count_reachable_combinations(
        coarse_cfg["num_routed"], coarse_cfg["top_k"], coarse_cfg.get("num_shared", 0)
    )
    fine_comb = count_reachable_combinations(
        fine_cfg["num_routed"], fine_cfg["top_k"], fine_cfg.get("num_shared", 0)
    )
    return {
        "coarse_combinations": coarse_comb,
        "fine_combinations": fine_comb,
        "ratio": float(fine_comb) / float(coarse_comb) if coarse_comb > 0 else 0.0
    }
