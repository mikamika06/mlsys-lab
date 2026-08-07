import sys
sys.path.insert(0, ".")
from streamllm.evict import evaluate_needle
from streamllm.cache import compute_perplexity
from streamllm.sweep import find_optimal_sink


def test_evaluate_needle_performance():
    score = evaluate_needle("h2o", 4096, 200)
    assert score > 0.5, f"needle retrieval score too low: {score}"


def test_compute_perplexity_bounds():
    val = compute_perplexity(4096, 4, 512, "sink_window")
    assert val < 10.0, f"perplexity too high: {val}"


def test_sweep_sinks_valid_choice():
    opt = find_optimal_sink(4096, 512, [0, 4, 8])
    assert opt in [0, 4, 8], f"invalid optimal sink: {opt}"
