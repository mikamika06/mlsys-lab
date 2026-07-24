from __future__ import annotations

import numpy as np


def add_vjp(a: np.ndarray, b: np.ndarray, grad_out: np.ndarray):
    """VJP of `out = a + b`, where `a` and `b` may have different but
    NumPy-broadcastable shapes.

    Parameters
    ----------
    a, b : np.ndarray
        Broadcastable arrays such that `a + b` is well-defined.
    grad_out : np.ndarray
        Upstream gradient dL/d(out), with shape `np.broadcast_shapes(a.shape, b.shape)`.

    Returns
    -------
    (grad_a, grad_b) : tuple[np.ndarray, np.ndarray]
        dL/da with shape `a.shape`, dL/db with shape `b.shape`. Any axis
        that broadcasting stretched (or added) must be summed back down
        ("sum-to-shape") -- do not just reshape or slice `grad_out`.
    """
    raise NotImplementedError('your code here')


def mul_vjp(a: np.ndarray, b: np.ndarray, grad_out: np.ndarray):
    """VJP of `out = a * b`, where `a` and `b` may have different but
    NumPy-broadcastable shapes.

    Parameters
    ----------
    a, b : np.ndarray
        Broadcastable arrays such that `a * b` is well-defined.
    grad_out : np.ndarray
        Upstream gradient dL/d(out), with shape `np.broadcast_shapes(a.shape, b.shape)`.

    Returns
    -------
    (grad_a, grad_b) : tuple[np.ndarray, np.ndarray]
        dL/da with shape `a.shape`, dL/db with shape `b.shape`.
    """
    raise NotImplementedError('your code here')
