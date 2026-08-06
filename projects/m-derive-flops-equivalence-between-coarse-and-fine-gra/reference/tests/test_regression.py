from moe.combinatorics import compare_combinations
from moe.flops import derive_fine_grained_split


def test_combination_expansion():
    coarse = {"num_routed": 8, "top_k": 2, "num_shared": 0}
    fine = {"num_routed": 32, "top_k": 6, "num_shared": 2}
    res = compare_combinations(coarse, fine)
    assert res["fine_combinations"] > res["coarse_combinations"]


def test_flops_parity():
    res = derive_fine_grained_split(
        d_model=512,
        d_ffn_coarse=2048,
        num_coarse=8,
        k_coarse=2,
        num_shared=2,
        k_fine=6,
        split_factor=4
    )
    assert res["is_equivalent"] is True
