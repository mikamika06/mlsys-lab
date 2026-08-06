import numpy as np


def apply_mask(weights, mask):
    raise NotImplementedError


def magnitude_mask(weights, sparsity):
    raise NotImplementedError


def iterative_prune(model, dataloader, num_rounds, sparsity_target, rewinding_weights):
    raise NotImplementedError
