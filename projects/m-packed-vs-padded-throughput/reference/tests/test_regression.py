import sys
sys.path.insert(0, ".")

from seqpack.simulate import throughput_ratio, misspecification_effects

def test_throughput_ratio_is_at_least_one():
    seqlens = [10, 50, 100, 15]
    ratio = throughput_ratio(seqlens, block_size=16)
    assert ratio >= 1.0, f"Expected ratio >= 1.0, got {ratio}"


def test_misspecification_penalty():
    seqlens = [10, 20, 30]
    eff_optimal = misspecification_effects(seqlens, 16, 30)
    assert eff_optimal["wasted_flops"] == 0
    assert eff_optimal["relative_degradation"] == 1.0

    eff_sub = misspecification_effects(seqlens, 16, 64)
    assert eff_sub["wasted_flops"] > 0
    assert eff_sub["relative_degradation"] > 1.0
