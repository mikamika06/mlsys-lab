import numpy as np

def magnitude_mask(W, sparsity):
    raise NotImplementedError("Implement magnitude pruning mask generation")

def wanda_mask(W, X, sparsity):
    raise NotImplementedError("Implement Wanda pruning mask generation")

def mask_recall(mask_a, mask_b):
    raise NotImplementedError("Calculate recall/overlap between two masks")
