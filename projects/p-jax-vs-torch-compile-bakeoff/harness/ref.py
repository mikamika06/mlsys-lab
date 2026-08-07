import numpy as np
from bakeoff.runner import BakeoffRunner
from bakeoff.models import StackModel

def check_structural_equivalence(cfg):
    m1 = StackModel(cfg)
    m2 = StackModel(cfg)
    x = np.ones((2, cfg.get("dim", 64)), dtype=np.float32)
    out1 = m1.forward(x)
    out2 = m2.forward(x)
    return np.allclose(out1, out2)

def check_metrics_and_ratio(cfg, inputs):
    runner = BakeoffRunner(cfg)
    ra = runner.compile_and_run("stack_a", inputs)
    rb = runner.compile_and_run("stack_b", inputs)
    valid = (ra["compilation_time"] > 0) and (rb["compilation_time"] > 0)
    ratio = ra["execution_time"] / (rb["execution_time"] + 1e-8)
    bounded = 0.1 <= ratio <= 10.0
    return valid, bounded

def check_dynamic_guard(cfg, shapes):
    runner = BakeoffRunner(cfg)
    res = runner.evaluate_dynamic("stack_a", shapes)
    return "recompilations" in res and isinstance(res["recompilations"], int)

def check_artifacts(cfg):
    runner = BakeoffRunner(cfg)
    art_a = runner.export_artifact("stack_a")
    art_b = runner.export_artifact("stack_b")
    match = ("format" in art_a) and ("format" in art_b)
    return match

def check_intervals_calc(runs_a, runs_b):
    runner = BakeoffRunner({"dim": 16})
    res = runner.compute_intervals(runs_a, runs_b)
    return "overlap" in res

def check_recommendation_logic():
    runner = BakeoffRunner({"dim": 16})
    r1 = runner.recommend("static_heavy")
    r2 = runner.recommend("dynamic_heavy")
    return r1 == "stack_a" and r2 == "stack_b"
