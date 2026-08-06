"""Regression tests for Unsloth benchmark utilities."""
from unsloth_bench.parser import parse_unsloth_log, compute_speedup_ratio


def test_parse_unsloth_log():
    log = """
    [Unsloth] Peak VRAM: 14.2 GB
    Step 100 | loss = 1.842
    Step 200 | 'loss': 1.125
    Training completed in 2.50 steps/s.
    """
    parsed = parse_unsloth_log(log)
    assert parsed["peak_vram_gb"] == 14.2
    assert parsed["steps_per_sec"] == 2.5
    assert parsed["final_loss"] == 1.125


def test_compute_speedup_ratio():
    ratio = compute_speedup_ratio(2.5, 1.25)
    assert ratio == 2.0
