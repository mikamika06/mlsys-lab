## Context

Reverse-mode automatic differentiation records every elementary operation
performed on scalar values as a node on a computation graph (a "Wengert
list"), each node remembering its inputs and a *local* backward rule. Given
an output node, gradients w.r.t. every ancestor are obtained by:

1. building a **topological order** of the graph (every node after all of
   its inputs), by depth-first search from the output;
2. seeding the output's gradient to $1$;
3. replaying each node's local backward rule in **reverse** topological
   order, accumulating (`+=`) into each input's `.grad`.

For a node $y = f(x_1, \dots, x_k)$ with upstream gradient
$\bar y = \partial L / \partial y$, the local rule pushes
$\bar y \cdot \partial y / \partial x_i$ into each $x_i$'s gradient. For the
four operations this task requires:

$$
y = a + b: \quad \bar a \mathrel{+}= \bar y, \quad \bar b \mathrel{+}= \bar y
$$

$$
y = a \cdot b: \quad \bar a \mathrel{+}= \bar y \cdot b, \quad
\bar b \mathrel{+}= \bar y \cdot a
$$

$$
y = \tanh(a): \quad \bar a \mathrel{+}= \bar y \cdot (1 - y^2)
$$

$$
y = \exp(a): \quad \bar a \mathrel{+}= \bar y \cdot y
$$

Because a value can feed into more than one downstream node (its gradient
is a **sum** over every path to the output), replaying in the wrong order —
or only once per node instead of accounting for every consumer — silently
corrupts gradients for any reused value. Reverse topological order is
exactly what guarantees a node's `.grad` is fully accumulated (from every
consumer) before that node's own `_backward` runs and pushes into *its*
inputs.

## Task

Implement a `Value` class, in the style of Karpathy's micrograd:

```python
class Value:
    def __init__(self, data):
        ...

    def __add__(self, other): ...
    __radd__ = __add__

    def __mul__(self, other): ...
    __rmul__ = __mul__

    def tanh(self): ...
    def exp(self): ...

    def backward(self): ...
```

- `Value(data)` wraps a Python scalar; `.data` holds the current value,
  `.grad` starts at `0.0`.
- `+` and `*` must work between two `Value`s, and between a `Value` and a
  plain `int`/`float` (in either operand order).
- `.tanh()` and `.exp()` return a new `Value` holding $\tanh(\text{self})$
  / $\exp(\text{self})$, wired into the graph.
- `out.backward()` sets `out.grad = 1.0` and propagates gradients to every
  ancestor of `out` reachable through `+`, `*`, `tanh`, `exp`, using
  correctly ordered reverse-mode replay as described above — including
  correct accumulation when a `Value` is used more than once.

## Example

```python
x = Value(2.0)
w = Value(-1.5)
b = Value(0.5)

z = w * x + b        # z.data == -2.5
h = z.tanh()          # h.data == tanh(-2.5)
h.backward()

# x.grad == d(tanh(w*x+b))/dx == w * (1 - h.data**2)
```

## What the gate checks

The grader builds a small two-hidden-unit network purely from `+`, `*`,
`tanh`, and `exp` — two inputs feed two `tanh` hidden units through shared
weights, whose outputs are linearly combined and passed through `exp` — and
runs `out.backward()`. This is checked against **two independent oracles**,
neither of which calls `sol.Value`:

- **`forward_max_abs_err`** — `out.data` compared to the same expression
  evaluated with plain `math.tanh` / `math.exp` on ordinary floats.
- **`max_abs_err`** — every leaf's `.grad` compared to a **central finite
  difference** of that same plain-float expression w.r.t. each leaf,
  $\partial f/\partial x_i \approx \big(f(x_i+h) - f(x_i-h)\big) / (2h)$,
  over several random parameter draws.

Both must hold to a tight tolerance; a wrong local backward rule, a missing
`+=` accumulation, or an incorrect (non-reverse) replay order all produce
gradient errors far above the gate's threshold.
