import sys
sys.path.insert(0, ".")
from pareto.front import select_pareto_front
from pareto.metrics import measure_retention
from pareto.compare import compare_checkpoints
import numpy as np


def test_select_pareto_basic():
    pts = [
        {"id": 1, "params": 100, "accuracy": 0.8},
        {"id": 2, "params": 50, "accuracy": 0.5},
        {"id": 3, "params": 80, "accuracy": 0.85}
    ]
    front = select_pareto_front(pts)
    assert len(front) == 2


def test_measure_retention_ratio():
    student_logits = np.array([[2.0, 1.0], [0.5, 1.5]])
    teacher_logits = np.array([[2.1, 0.9], [0.4, 1.6]])
    targets = np.array([0, 1])
    ret = measure_retention(student_logits, teacher_logits, targets)
    assert 0.0 <= ret <= 2.0


def test_compare_checkpoints_validity():
    cps = [
        {"id": "a", "params": 10, "accuracy": 0.5},
        {"id": "b", "params": 20, "accuracy": 0.6}
    ]
    res = compare_checkpoints(cps)
    assert len(res) == 2
