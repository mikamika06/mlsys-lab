import sys

sys.path.insert(0, ".")
from quant_recipes.allocator import optimal_alloc, uniform_alloc

def test_excluded_layers_are_kept_at_base_bits():
    profile = [
        {"name": "head", "params": 100, "sens": {8: 0.1, 4: 10.0}},
        {"name": "L1", "params": 100, "sens": {8: 0.1, 4: 10.0}}
    ]
    ans = optimal_alloc(profile, ["head"], 2400, 16)
    assert ans["head"] == 16

def test_optimal_alloc_stays_within_budget():
    profile = [{"name": "L1", "params": 100, "sens": {8: 0.1, 4: 10.0}}]
    ans = optimal_alloc(profile, [], 400, 16)
    assert ans["L1"] == 4
