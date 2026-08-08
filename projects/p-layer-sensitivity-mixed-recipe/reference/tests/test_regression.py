import sys

sys.path.insert(0, ".")
import quant


def test_budget_honored():
    shapes = {"l1": (32, 32), "l2": (32, 64)}
    sens = {"l1": {8: 0.0, 4: 1.0, 2: 5.0}, "l2": {8: 0.0, 4: 0.5, 2: 2.0}}
    budget = 800
    recipe = quant.build_recipe(shapes, sens, budget, [8, 4, 2])
    size = quant.get_size_bytes(shapes["l1"], recipe["l1"]) + quant.get_size_bytes(shapes["l2"], recipe["l2"])
    assert size <= budget
