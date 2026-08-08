"""Bit budget allocation engine."""

import numpy as np


def compute_recipe_size_bytes(recipe: dict, layer_shapes: dict) -> int:
    """Calculate model total weight size in bytes for given layer bitwidth recipe."""
    raise NotImplementedError


def build_recipe(sensitivity_map: dict, layer_shapes: dict, budget_bytes: int, candidate_bits: list[int]) -> dict:
    """
    Construct optimal layer-to-bitwidth recipe adhering to size budget.
    Returns dict: {layer_name: bitwidth}
    """
    raise NotImplementedError


def find_uniform_recipe(layer_shapes: dict, budget_bytes: int, candidate_bits: list[int]) -> dict:
    """Construct uniform precision recipe that fits within size budget."""
    raise NotImplementedError
