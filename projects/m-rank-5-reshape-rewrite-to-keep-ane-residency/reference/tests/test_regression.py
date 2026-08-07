import sys

sys.path.insert(0, ".")
from anerewrite.rewrite import rewrite_shape
from anerewrite.score import compute_residency_score
from anerewrite.predict import predict_ane_friendly


def test_rewrite_rank_lte_5():
    res = rewrite_shape([1, 2, 3, 4, 5, 6])
    assert len(res) <= 5


def test_score_range():
    plan = {"ops": [{"target": "ANE", "cost": 10}, {"target": "CPU", "cost": 10}]}
    s = compute_residency_score(plan)
    assert 0.0 <= s <= 1.0


def test_prediction_prefers_ane_friendly():
    res = predict_ane_friendly({"max_rank": 2, "reshapes": 1}, {"max_rank": 8, "reshapes": 10})
    assert res == "arch_a"
