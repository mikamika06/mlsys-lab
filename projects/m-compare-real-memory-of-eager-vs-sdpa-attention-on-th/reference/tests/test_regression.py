import torch
import pytest
from attnmem.model import TinyAttentionModel
from attnmem.measure import compute_size_ratio


def test_attention_memory_behavior():
    config = {"hidden_size": 64, "num_heads": 4}
    model = TinyAttentionModel(config)
    inputs = torch.randn(2, 32, 64)
    ratio = compute_size_ratio(model, inputs)
    assert ratio > 1.0
