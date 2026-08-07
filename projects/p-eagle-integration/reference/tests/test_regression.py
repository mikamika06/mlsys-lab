import sys
sys.path.insert(0, ".")
from eagle.sampler import DraftSampler
from eagle.head import DraftHead
import numpy as np


def test_temperature_scaling():
    sampler_cold = DraftSampler(temperature=0.1)
    sampler_hot = DraftSampler(temperature=2.0)
    logits = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    s_cold = sampler_cold.sample(logits)
    s_hot = sampler_hot.sample(logits)
    assert s_cold == 1
    assert isinstance(s_hot, int)


def test_head_forward():
    head = DraftHead(hidden_dim=4, vocab_size=10)
    hs = [1.0, 0.0, -1.0, 0.5]
    out = head.forward(hs)
    assert out.shape == (10,)
