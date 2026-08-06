import numpy as np


def check_acceptance(p, q, token, u):
    return p[token] >= q[token] * u


def argmin_index(array_like):
    return int(np.argmin(array_like))
