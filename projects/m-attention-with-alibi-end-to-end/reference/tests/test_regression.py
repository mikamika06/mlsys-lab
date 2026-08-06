import numpy as np
from alibi_attn.overflow import measure_overflow_rate

def test_alibi_attention_overflow():
    scores = np.array([100000.0, -100000.0, 70000.0], dtype=np.float32)
    uncapped_rate = measure_overflow_rate(scores, threshold=65504.0, softcap=None)
    assert uncapped_rate > 0.0

    capped_rate = measure_overflow_rate(scores, threshold=65504.0, softcap=50.0)
    assert capped_rate == 0.0
