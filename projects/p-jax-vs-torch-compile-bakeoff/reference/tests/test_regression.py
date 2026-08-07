import sys
import numpy as np

sys.path.insert(0, ".")
from bakeoff.runner import BakeoffRunner

def test_bakeoff_basic_execution():
    runner = BakeoffRunner({"dim": 16})
    res_a = runner.compile_and_run("stack_a", [np.zeros((4, 16), dtype=np.float32)])
    res_b = runner.compile_and_run("stack_b", [np.zeros((4, 16), dtype=np.float32)])
    assert res_a["execution_time"] >= 0.0
    assert res_b["execution_time"] >= 0.0

def test_dynamic_shapes_behavior():
    runner = BakeoffRunner({"dim": 16})
    eval_res = runner.evaluate_dynamic("stack_a", [(4, 16), (4, 16), (8, 16)])
    assert "recompilations" in eval_res

def test_recommendation_logic():
    runner = BakeoffRunner({"dim": 16})
    rec = runner.recommend("static_heavy")
    assert rec == "stack_a"
