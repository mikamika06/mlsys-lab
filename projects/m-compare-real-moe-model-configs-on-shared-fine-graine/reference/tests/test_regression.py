from moecomp.metrics import compute_costs

def test_cost_calculation_invariants():
    cfg = {
        "hidden_size": 256,
        "moe_intermediate_size": 512,
        "num_experts": 8,
        "num_shared_experts": 2,
        "fine_grained_factor": 2
    }
    res = compute_costs(cfg)
    assert res["total_params"] > 0
    assert res["shared_params"] >= 0
    assert res["routed_params"] >= 0

def test_effective_experts_scaling():
    cfg = {
        "hidden_size": 128,
        "moe_intermediate_size": 256,
        "num_experts": 16,
        "num_shared_experts": 1,
        "fine_grained_factor": 4
    }
    res = compute_costs(cfg)
    assert res["effective_experts"] == 4
