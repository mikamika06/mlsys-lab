import sys

sys.path.insert(0, ".")
from profiler_metrics.overhead import compute_throughput_ratio
from profiler_metrics.ranking import rank_profiler_flags


def test_throughput_ratio_positive():
    base = [100.0, 200.0, 300.0]
    cp = [50.0, 100.0, 150.0]
    ps = [90.0, 180.0, 270.0]
    ratio = compute_throughput_ratio(base, cp, ps)
    assert ratio > 0.0, f"Expected positive ratio, got {ratio}"


def test_flag_ranking_order():
    measurements = {
        "record_shapes": 0.15,
        "with_stack": 0.45,
        "profile_memory": 0.25,
        "with_flops": 0.05
    }
    ranked = rank_profiler_flags(measurements)
    assert ranked[0] == "with_flops", f"Expected lowest overhead first, got {ranked}"
    assert ranked[-1] == "with_stack", f"Expected highest overhead last, got {ranked}"


def test_ranking_length():
    measurements = {
        "record_shapes": 0.1,
        "with_stack": 0.3,
        "profile_memory": 0.2,
        "with_flops": 0.05
    }
    ranked = rank_profiler_flags(measurements)
    assert len(ranked) == 4
