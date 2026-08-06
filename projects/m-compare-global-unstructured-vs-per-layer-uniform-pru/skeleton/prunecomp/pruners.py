"""Pruning mask and threshold utilities."""

import numpy as np


def compute_per_layer_uniform_masks(weights, sparsity_ratio):
    """Compute per-layer uniform pruning masks."""
    raise NotImplementedError


def compute_global_unstructured_mask(weights, sparsity_ratio):
    """Compute global unstructured pruning mask."""
    raise NotImplementedError
