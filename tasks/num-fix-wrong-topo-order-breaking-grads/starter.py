"""Buggy reverse-mode scalar autograd — find and fix the bug.

`Value.backward()` builds a topological order of the tape and is supposed
to replay each node's local `_backward` closure in REVERSE topological
order. This version builds the same topological order but forgets to
reverse it before replaying, so a node's gradient can be "closed" (its
`_backward` called, pushing into its own children) before every consumer
of that node has finished adding its contribution to `.grad`. Any value
that is reused more than once in the graph ends up with a wrong, usually
zeroed-out, gradient.
"""
from __future__ import annotations


class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in topo:          # BUG: should be `reversed(topo)`
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
