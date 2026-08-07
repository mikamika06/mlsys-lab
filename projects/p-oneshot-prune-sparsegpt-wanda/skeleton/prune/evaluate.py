import numpy as np
from prune.methods import prune_layer


def eval_loss(W, W_pruned, X):
    raise NotImplementedError


def run_benchmark(W, X, sparsity):
    raise NotImplementedError
