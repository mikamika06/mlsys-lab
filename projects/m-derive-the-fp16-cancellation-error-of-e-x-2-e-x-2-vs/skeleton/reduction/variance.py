import numpy as np

def one_pass_variance(x):
    """
    Computes variance using E[x^2] - E[x]^2.
    Must perform all intermediate math strictly in float16.
    """
    raise NotImplementedError

def two_pass_variance(x):
    """
    Computes variance using E[(x - E[x])^2].
    Must perform all intermediate math strictly in float16.
    """
    raise NotImplementedError
