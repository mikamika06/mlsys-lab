import sys

sys.path.insert(0, ".")

from checkpointing.sim import simulate_checkpointing, baseline, optimal_segments


def test_checkpointing_peak_memory_includes_recomputation():
    res = simulate_checkpointing(10, 2, 100, 10, 20)
    assert res["peak_mem"] == 600, "Should include 2 checkpoints (200) + 4 intermediates (400)"


def test_baseline_memory_is_linear():
    res = baseline(10, 100, 10, 20)
    assert res["peak_mem"] == 1000


def test_optimal_segments_beats_baseline():
    b = baseline(20, 100, 10, 20)
    opt_s = optimal_segments(20)
    c = simulate_checkpointing(20, opt_s, 100, 10, 20)
    assert c["peak_mem"] < b["peak_mem"]
    assert c["step_time"] > b["step_time"]
