import torch


def true_branch(x):
    return x * 2.0


def false_branch(x):
    return x * 3.0


def conditional_branch_fn(x):
    pred = x.sum() > 0
    return torch.cond(pred, true_branch, false_branch, [x])
