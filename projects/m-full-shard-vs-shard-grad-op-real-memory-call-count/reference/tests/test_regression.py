import sys
sys.path.insert(0, ".")
from fsdp_analysis.model import compute_costs
from fsdp_analysis.memory import predict_fsdp_units
from fsdp_analysis.optimal import optimal_wrap_granularity

def test_strategy_memory_relation():
    full = compute_costs("FULL_SHARD", 1000000, 512, 12, 8)
    grad = compute_costs("SHARD_GRAD_OP", 1000000, 512, 12, 8)
    assert full["peak_memory"] < grad["peak_memory"], "FULL_SHARD memory should be lower than SHARD_GRAD_OP"

def test_unit_prediction_bounds():
    units = predict_fsdp_units(24, 5000000, 1000000)
    assert units > 0
    assert units <= 24

def test_optimal_granularity_validity():
    res = optimal_wrap_granularity(100000000, 8, 0.01, 1024 * 1024 * 1024)
    assert "optimal_units" in res
    assert res["optimal_units"] > 0
