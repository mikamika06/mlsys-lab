## Context

A reverse-mode scalar autograd engine (à la micrograd) records every
operation on a `Value` as it happens — a **Wengert list** (tape). Each
node $v$ stores its inputs `_prev(v)` and a local backward rule that,
given $\partial L/\partial v$, adds the right contribution to
$\partial L/\partial u$ for every $u \in \mathrm{prev}(v)$:

$$
\frac{\partial L}{\partial u} \mathrel{+}= \frac{\partial v}{\partial u}\cdot\frac{\partial L}{\partial v}.
$$

This accumulation is only correct if, by the time $v$'s own local rule
runs (pushing gradient further into $\mathrm{prev}(v)$), $v$'s gradient
has **already received every contribution from all of its consumers**.
That is precisely the guarantee a **reverse topological order** gives:
process the root first, and only ever process a node after every node
that used it as an input has already been processed.

## Task

The `Value` class below builds the topological order of the tape
correctly (a standard post-order DFS: every child appears before its
parent), but `backward()` replays it in the wrong order, so nodes that
are reused more than once in the graph never accumulate their full
gradient before propagating it onward.

```python
class Value:
    def __init__(self, data, _children=()): ...
    def __add__(self, other): ...
    def __mul__(self, other): ...
    def backward(self): ...
```

* `Value(data)` — leaf node, `data` a Python float.
* `+`, `*` — build new `Value`s, recording `_prev` and a local backward
  closure; both accept a `Value` or a plain number on the right side.
* `v.backward()` — sets `v.grad = 1.0` and propagates gradients into
  every ancestor `Value` reachable from `v`, accumulating into `.grad`
  with `+=` wherever a node is reused.

Find and fix the ordering bug in `backward()`.

## Example

```python
x0, x1, x2 = Value(2.0), Value(3.0), Value(-1.0)
a = x0 * x1          # a is reused below
b = a + x2
out = a * b + a      # a appears three times in the graph

out.backward()
print(x0.grad, x1.grad, x2.grad)
```

With the bug, `x0.grad`, `x1.grad`, `x2.grad` come out wrong (typically
zero) because `a`'s local backward rule fires before all three of its
uses have added their contribution to `a.grad`. With the fix, they match
the true gradient of `out` with respect to `x0`, `x1`, `x2`.

## What the gate checks

The grader builds a small diamond-shaped expression graph out of
`sol.Value` objects (some leaves reused three times, at different
depths), calls `.backward()`, and compares the resulting leaf `.grad`
values against a **central finite-difference** gradient of an ordinary,
autograd-free Python function that computes the exact same expression.
This is repeated for 6 random input vectors.

* **rel_err** — mean relative L2 error between your analytic gradients
  and the finite-difference gradients, across all 6 trials. Must be
  below `1e-6`.
