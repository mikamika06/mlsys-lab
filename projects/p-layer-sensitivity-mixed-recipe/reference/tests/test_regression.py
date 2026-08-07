import sys
sys.path.insert(0, ".")
from quant.sensitivity import measure_sensitivity
from quant.recipe import build_recipe
from quant.mixed import apply_mixed_quantization, evaluate_model
import numpy as np

def test_recipe_size():
    model = {"l1": np.array([[1.0, 0.5], [0.2, 1.0]]), "l2": np.array([[0.5, 0.1], [0.1, 0.5]])}
    dl = [(np.array([[1.0, 0.0]]), np.array([[1.0, 0.5]])) for _ in range(5)]
    sens = measure_sensitivity(model, dl)
    recipe = build_recipe(sens, budget_bits=4, allowed_bits=[2, 4])
    assert len(recipe) == len(model)

def test_mixed_improves_over_uniform():
    model = {"l1": np.array([[1.0, 0.2], [0.1, 1.0]]), "l2": np.array([[0.5, 0.2], [0.2, 0.5]])}
    dl = [(np.array([[1.0, 1.0]]), np.array([[1.2, 1.2]])) for _ in range(5)]
    sens = measure_sensitivity(model, dl)
    recipe = build_recipe(sens, budget_bits=3, allowed_bits=[2, 4])
    q_model = apply_mixed_quantization(model, recipe)
    loss = evaluate_model(q_model, dl)
    assert loss >= 0.0
