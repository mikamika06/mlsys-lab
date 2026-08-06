import torch
from lora.counts import count_parameters


def test_parameter_counts_positive():
    counts = count_parameters(768, 768, 8)
    assert counts["total"] > 0
    assert counts["lora_a"] == 8 * 768
    assert counts["lora_b"] == 768 * 8
