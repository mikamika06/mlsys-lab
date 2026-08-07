import numpy as np


def compute_importance(W, X, method="wanda"):
    raise NotImplementedError


def prune_layer(W, X, sparsity=0.5, method="wanda"):
    raise NotImplementedError


def evaluate_model(weights, inputs, targets):
    raise NotImplementedError


def compare_methods(weights, inputs, targets, sparsity=0.5):
    raise NotImplementedError


def check_loss_bound(baseline_loss, pruned_loss, max_ratio=1.5):
    raise NotImplementedError


def generate_report(results):
    raise NotImplementedError
