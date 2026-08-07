import sys
sys.path.insert(0, ".")
from spec.concurrency import decide_speculation

def test_speculation_disabled_at_high_concurrency_low_acceptance():
    model = {"base_latency": 10.0, "draft_latency": 2.0}
    assert decide_speculation(64, 0.1, model) is False

def test_speculation_enabled_at_low_concurrency_high_acceptance():
    model = {"base_latency": 10.0, "draft_latency": 2.0}
    assert decide_speculation(1, 0.9, model) is True
