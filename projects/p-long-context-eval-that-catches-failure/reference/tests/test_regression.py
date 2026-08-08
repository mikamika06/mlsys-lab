import sys
sys.path.insert(0, ".")
from longctx.generator import generate_tasks
from longctx.evaluator import evaluate_curve, detect_dip
from longctx.analyzer import separate_failures, compare_methods
import numpy as np

def test_generation_coverage():
    tasks = generate_tasks(1000, 5)
    assert len(tasks) == 5
    assert all(0.0 <= t["position"] <= 1.0 for t in tasks)

def test_dip_detection():
    tasks = generate_tasks(2000, 11)
    curve = evaluate_curve(tasks, "flawed")
    assert detect_dip(curve) == True

def test_healthy_model_no_dip():
    tasks = generate_tasks(2000, 11)
    curve = evaluate_curve(tasks, "healthy")
    assert detect_dip(curve) == False

def test_failure_separation():
    res = separate_failures(np.array([0.02]), np.array([1, 2]))
    assert res == "attention_failure"

def test_method_comparison():
    res = compare_methods([], [0.2, 0.3], [0.9, 0.95])
    assert res["superior"] == "method_b"
