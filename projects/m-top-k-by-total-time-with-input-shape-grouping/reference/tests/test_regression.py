import sys

sys.path.insert(0, ".")
from analyzer.profiler import aggregate_by_shape, top_k_by_total_time


def test_shapes_are_not_merged():
    events = [
        {"name": "aten::mm", "ph": "X", "dur": 100, "args": {"Input Dims": [[32, 128], [128, 64]]}},
        {"name": "aten::mm", "ph": "X", "dur": 200, "args": {"Input Dims": [[64, 128], [128, 32]]}}
    ]
    agg = aggregate_by_shape(events)
    assert len(agg) == 2, "Shapes were incorrectly merged into a single key"


def test_top_k_preserves_shapes():
    events = [
        {"name": "aten::mm", "ph": "X", "dur": 100, "args": {"Input Dims": [[32, 32]]}},
        {"name": "aten::mm", "ph": "X", "dur": 200, "args": {"Input Dims": [[64, 64]]}}
    ]
    top = top_k_by_total_time(events, k=2)
    assert len(top) == 2
    assert top[0]["shapes"] == ((64, 64),)
    assert top[1]["shapes"] == ((32, 32),)
