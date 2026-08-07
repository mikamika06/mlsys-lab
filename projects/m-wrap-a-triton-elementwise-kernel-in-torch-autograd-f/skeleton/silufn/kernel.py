import torch


def forward_kernel(x):
    raise NotImplementedError


def backward_kernel(x, grad_output):
    raise NotImplementedError
