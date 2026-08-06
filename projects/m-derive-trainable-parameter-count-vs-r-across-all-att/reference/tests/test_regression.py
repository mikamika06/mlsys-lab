from loraspec.scaling import compute_scaling_factor


def test_lora_scaling_invariance():
    s1 = compute_scaling_factor(alpha=16.0, r=8, mode="lora")
    s2 = compute_scaling_factor(alpha=16.0, r=16, mode="lora")
    assert s1 == 2.0
    assert s2 == 1.0
    assert s1 / s2 == 2.0
