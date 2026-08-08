"""Model evaluation and comparison engine."""

import numpy as np
from mixed_quant.sensitivity import quantize_weight
from mixed_quant.allocator import build_recipe, find_uniform_recipe, compute_recipe_size_bytes
from mixed_quant.sensitivity import measure_layer_sensitivity


def forward_model(model: dict, dataset: np.ndarray, recipe: dict) -> np.ndarray:
    """Run model forward pass applying recipe bitwidths to model weights."""
    x = dataset
    for name, w in model["layers"].items():
        b = recipe.get(name, 16)
        w_q = quantize_weight(w, b)
        x = np.matmul(x, w_q)
        x = np.maximum(0, x)
    return x


def evaluate_recipe(model: dict, dataset: np.ndarray, target: np.ndarray, recipe: dict) -> float:
    """Evaluate end-to-end MSE loss for model under specific recipe."""
    pred = forward_model(model, dataset, recipe)
    return float(np.mean((pred - target) ** 2))


def compare_recipes(model: dict, dataset: np.ndarray, target: np.ndarray, layer_shapes: dict, budget_bytes: int, candidate_bits: list[int]) -> dict:
    """Compare mixed-precision recipe against uniform recipe under same budget constraint."""
    sens = measure_layer_sensitivity(model, dataset, candidate_bits)
    mixed_recipe = build_recipe(sens, layer_shapes, budget_bytes, candidate_bits)
    uniform_recipe = find_uniform_recipe(layer_shapes, budget_bytes, candidate_bits)

    mixed_loss = evaluate_recipe(model, dataset, target, mixed_recipe)
    uniform_loss = evaluate_recipe(model, dataset, target, uniform_recipe)

    mixed_size = compute_recipe_size_bytes(mixed_recipe, layer_shapes)
    uniform_size = compute_recipe_size_bytes(uniform_recipe, layer_shapes)

    return {
        "mixed_recipe": mixed_recipe,
        "uniform_recipe": uniform_recipe,
        "mixed_loss": mixed_loss,
        "uniform_loss": uniform_loss,
        "mixed_size_bytes": mixed_size,
        "uniform_size_bytes": uniform_size,
        "mixed_beats_uniform": mixed_loss <= uniform_loss,
    }


def verify_budget_and_loss(model: dict, dataset: np.ndarray, target: np.ndarray, recipe: dict, layer_shapes: dict, budget_bytes: int, max_loss: float) -> dict:
    """Verify size budget compliance and loss threshold constraint."""
    size = compute_recipe_size_bytes(recipe, layer_shapes)
    loss = evaluate_recipe(model, dataset, target, recipe)
    size_ok = size <= budget_bytes
    loss_ok = loss <= max_loss
    return {
        "size_bytes": size,
        "loss": loss,
        "size_within_budget": size_ok,
        "loss_within_bound": loss_ok,
        "passed": size_ok and loss_ok,
    }
