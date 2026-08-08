import sys

sys.path.insert(0, ".")
from mixplan.recipe import recipe_bytes
from mixplan.solver import solve_recipe
from mixplan.verify import verify_f32_1d

CONFIG = {
    "tensors": [
        {"name": "blk.0.attn_q.weight", "shape": [4096, 4096], "importance": 0.9},
        {"name": "blk.0.attn_norm.weight", "shape": [4096], "importance": 1.0},
        {"name": "blk.0.ffn_down.weight", "shape": [4096, 11008], "importance": 0.7},
    ],
    "budget_bytes": 100000000,
}


def test_recipe_hits_budget():
    recipe = solve_recipe(CONFIG, CONFIG["budget_bytes"])
    assert recipe_bytes(CONFIG, recipe) <= CONFIG["budget_bytes"]


def test_1d_tensors_remain_f32():
    recipe = solve_recipe(CONFIG, CONFIG["budget_bytes"])
    assert verify_f32_1d(CONFIG, recipe) is True


def test_invalid_1d_quantization_fails_verification():
    recipe = solve_recipe(CONFIG, CONFIG["budget_bytes"])
    recipe["blk.0.attn_norm.weight"] = "Q8_0"
    assert verify_f32_1d(CONFIG, recipe) is False
