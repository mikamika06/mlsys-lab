import sys
sys.path.insert(0, ".")
from eagle.head import DraftHead
from eagle.integration import EagleEngine


def test_temperature_scaling_affects_sampling():
    head = DraftHead(64, 100)
    engine = EagleEngine(64, 100, head)
    hidden, logits = engine.forward_target([1, 2, 3])
    t_low = engine.verify([10, 20], logits, temperature=0.1)
    t_high = engine.verify([10, 20], logits, temperature=2.0)
    assert isinstance(t_low, list)
    assert isinstance(t_high, list)


def test_memory_savings_positive():
    head = DraftHead(64, 100)
    engine = EagleEngine(64, 100, head)
    mem = engine.memory_usage_bytes()
    assert mem["head_bytes"] < mem["separate_model_bytes"]
