import sys
sys.path.insert(0, ".")
from ortprof.ranking import rank_top_slowest
from ortprof.overhead import compute_overhead
from ortprof.category import classify_categories


def test_ranking_length():
    events = [
        {"cat": "Node", "args": {"op_name": "MatMul"}, "dur": 100},
        {"cat": "Node", "args": {"op_name": "Add"}, "dur": 50},
        {"cat": "Node", "args": {"op_name": "Mul"}, "dur": 25},
        {"cat": "Node", "args": {"op_name": "Relu"}, "dur": 10},
        {"cat": "Node", "args": {"op_name": "Sub"}, "dur": 5},
        {"cat": "Node", "args": {"op_name": "Div"}, "dur": 2},
    ]
    top = rank_top_slowest(events)
    assert len(top) <= 5
    assert top[0] == "MatMul"


def test_overhead_bounds():
    events = [
        {"cat": "Profiling", "name": "Prof", "dur": 20},
        {"cat": "Node", "args": {"op_name": "MatMul"}, "dur": 80}
    ]
    oh = compute_overhead(events)
    assert 0.0 <= oh <= 1.0
    assert abs(oh - 0.2) < 1e-6


def test_category_shares_sum_to_one():
    events = [
        {"cat": "Node", "args": {"op_name": "MatMul"}, "dur": 60},
        {"cat": "Node", "args": {"op_name": "Add"}, "dur": 40}
    ]
    shares = classify_categories(events)
    total = sum(shares.values())
    assert abs(total - 1.0) < 1e-6
    assert shares["GEMM"] == 0.6
    assert shares["Elementwise"] == 0.4
