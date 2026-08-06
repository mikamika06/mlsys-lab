import numpy as np


def compute_moe_ffn_flops(d_model, d_ffn, num_tokens):
    return 3 * 2 * d_model * d_ffn * num_tokens


def derive_fine_grained_split(d_model, d_ffn_coarse, num_coarse, k_coarse, num_shared, k_fine, split_factor):
    d_ffn_fine = int(d_ffn_coarse // split_factor)
    flops_coarse_per_token = k_coarse * compute_moe_ffn_flops(d_model, d_ffn_coarse, 1)
    flops_fine_routed_per_token = k_fine * compute_moe_ffn_flops(d_model, d_ffn_fine, 1)
    flops_shared_per_token = num_shared * compute_moe_ffn_flops(d_model, d_ffn_fine, 1)
    flops_fine_total_per_token = flops_fine_routed_per_token + flops_shared_per_token

    return {
        "d_ffn_fine": d_ffn_fine,
        "coarse_flops_per_token": flops_coarse_per_token,
        "fine_flops_per_token": flops_fine_total_per_token,
        "is_equivalent": flops_coarse_per_token == flops_fine_total_per_token
    }
