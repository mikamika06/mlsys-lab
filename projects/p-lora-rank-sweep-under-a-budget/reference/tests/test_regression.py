import sys
sys.path.insert(0, ".")
from lora_sweep import config, engine, optimizer

def test_config_valid():
    cfg = config.get_default_config()
    assert cfg["budget_hours"] == 6.0

def test_baseline_eval():
    cfg = config.get_default_config()
    res = engine.run_baseline(cfg)
    assert res["loss"] <= 2.5

def test_rank_sweep():
    res = engine.run_rank_sweep([4, 8, 16], 500)
    assert len(res) == 3

def test_modules():
    res = optimizer.evaluate_modules([["q_proj"], ["q_proj", "v_proj"]])
    assert len(res) == 2

def test_alpha():
    res = optimizer.analyze_alpha_scaling([16, 32], [8, 16])
    assert "optimal_alpha" in res

def test_pareto():
    dummy = [{"rank": 8, "loss": 2.1, "cost": 100}, {"rank": 16, "loss": 2.0, "cost": 150}]
    res = optimizer.find_pareto_front(dummy)
    assert len(res) > 0

def test_second_domain():
    cfg = config.get_default_config()
    res = optimizer.verify_second_domain(cfg)
    assert res["verified"] is True
