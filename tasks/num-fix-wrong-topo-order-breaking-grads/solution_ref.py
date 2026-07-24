"""Reference solution for `num-fix-wrong-topo-order-breaking-grads`.

A minimal reverse-mode scalar autograd node (Wengert-list style), modeled
after micrograd. Each op records its inputs (`_prev`) and a local
`_backward` closure; `Value.backward()` builds a topological order of the
tape and replays the local backward closures in REVERSE topological order.
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
        # Build a topological order of the tape: every child appears
        # before its parent (post-order DFS).
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
        # A node's local `_backward` may only run once ALL of its
        # consumers have already added their contribution to its `.grad`.
        # That is exactly reverse topological order: the root (which has
        # no consumers) goes first, leaves go last.
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
