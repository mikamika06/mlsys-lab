import torch


def true_branch_mismatch(x):
    return x


def false_branch_mismatch(x):
    return x.unsqueeze(0)


def trigger_mismatch(x):
    pred = x.sum() > 0
    return torch.cond(pred, true_branch_mismatch, false_branch_mismatch, [x])
