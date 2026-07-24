"""Micrograd-style scalar autograd: Value with +, *, tanh, exp, backward()."""
from __future__ import annotations


class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        """Return a new Value = self + other, wired for backward()."""
        raise NotImplementedError('your code here')

    __radd__ = __add__

    def __mul__(self, other):
        """Return a new Value = self * other, wired for backward()."""
        raise NotImplementedError('your code here')

    __rmul__ = __mul__

    def tanh(self):
        """Return a new Value = tanh(self), wired for backward()."""
        raise NotImplementedError('your code here')

    def exp(self):
        """Return a new Value = exp(self), wired for backward()."""
        raise NotImplementedError('your code here')

    def backward(self):
        """Seed self.grad = 1.0 and propagate gradients to every ancestor
        in reverse topological order over the recorded tape."""
        raise NotImplementedError('your code here')
