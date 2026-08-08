"""Model evaluation and comparison engine."""

import numpy as np


def forward_model(model: dict, dataset: np.ndarray, recipe: dict) -> np.ndarray:
    """Run model forward pass applying recipe bitwidths to model weights."""
    raise NotImplementedError


def evaluate_recipe(model: dict, dataset: np.ndarray, target: np.ndarray, recipe: dict) -> float:
    """Evaluate end-to-end MSE loss for model under specific recipe."""
    raise NotImplementedError


def compare_recipes(model: dict, dataset: np.ndarray, target: np.ndarray, layer_shapes: dict, budget_bytes: int, candidate_bits: list[int]) -> dict:
    """Compare mixed-precision recipe against uniform recipe under same budget constraint."""
    raise NotImplementedError


def verify_budget_and_loss(model: dict, dataset: np.ndarray, target: np.ndarray, recipe: dict, layer_shapes: dict, budget_bytes: int, max_loss: float) -> dict:
    """Verify size budget compliance and loss threshold constraint."""
    raise NotImplementedError
