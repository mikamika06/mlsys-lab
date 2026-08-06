from optmem.gap import memory_multiplier_gap


def test_multiplier_positive():
    assert memory_multiplier_gap(1000, 100) == 10.0


def test_multiplier_zero_lora():
    assert memory_multiplier_gap(1000, 0) == 0.0
