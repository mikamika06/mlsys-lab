## Context

Reverse-mode autograd records a computation as a **Wengert list**: a flat
sequence of elementary operations, each one consuming values produced
earlier. Because every operation only reads nodes that were computed
*before* it, the list is already in topological order — the forward pass is
simply "run the list top to bottom."

The backward pass runs the list in **reverse**, propagating an *adjoint*
(the partial derivative of the final output with respect to each node) from
the output back to the inputs. For a node $z = \mathrm{op}(a, b)$ with
adjoint $\bar z = \partial \text{output}/\partial z$, the chain rule pushes
contributions onto its inputs:

$$
\bar a \mathrel{+}= \bar z \cdot \frac{\partial z}{\partial a}, \qquad
\bar b \mathrel{+}= \bar z \cdot \frac{\partial z}{\partial b}.
$$

The `+=` is the crux of this task. If a node's value is used by **more than
one** downstream operation — a shared subexpression — its adjoint is the
**sum** of every contribution it receives, not just the last one written.
For example, in

$$
t = x_0 x_1, \qquad f = \sin(t) + t\,x_2 + t
$$

the node $t$ feeds three different consumers, so

$$
\frac{\partial f}{\partial t} = \cos(t) + x_2 + 1,
$$

and a backward pass that *overwrites* $\bar t$ instead of accumulating it
will silently keep only one of these three terms.

## Task

Implement `tape_grad`:

```python
def tape_grad(tape: list[tuple[str, tuple[int, ...]]], x: list[float]) -> list[float]:
    ...
```

* `tape` — a Wengert list: a list of `(op, input_node_indices)` entries, one
  per computed node, already given in topological (forward-evaluation)
  order. Node indices `0 .. len(x)-1` refer to the entries of `x`; node
  index `len(x) + i` refers to the value produced by `tape[i]`. Supported
  ops: `"add"`, `"sub"`, `"mul"` (each with a 2-tuple of input indices) and
  `"sin"` (with a 1-tuple). The output of the whole computation is the value
  of the **last** node in the tape.
* `x` — list of floats, the input values.

Run a forward pass to obtain every node's value, then a backward pass over
the tape **in reverse order**, seeding the output node's adjoint with `1.0`
and propagating adjoints to each op's inputs using the local derivative
rules above — accumulating (`+=`) whenever a node is an input to more than
one op. Return the adjoints of the first `len(x)` nodes (i.e. the gradient
with respect to `x`) as a 1-D array of shape `(len(x),)`.

## Example

```python

# t = x0*x1 ; f = sin(t) + t*x2 + t
tape = [
    ("mul", (0, 1)),   # node 3 = x0*x1
    ("sin", (3,)),     # node 4 = sin(node 3)
    ("mul", (3, 2)),   # node 5 = node 3 * x2
    ("add", (4, 5)),   # node 6 = node 4 + node 5
    ("add", (6, 3)),   # node 7 = node 6 + node 3   <- output
]
x = [0.6, -0.4, 1.1]
grad = tape_grad(tape, x)
```

`grad[0]` must equal $\partial f/\partial x_0$, matching the analytic
expression $\cos(t)\,x_1 + x_1 x_2 + x_1$ where $t = x_0 x_1$ — every one of
the three paths through the reused node $t$ contributes.

## What the gate checks

A single gate named **rel_err** runs your implementation on two different
tapes (each with a node feeding multiple downstream ops) at several random
input points, and compares the returned gradient against a **central
finite-difference** estimate of the same scalar function — a completely
independent, real numerical oracle. The threshold is $10^{-5}$: a correct
reverse-mode implementation matches finite differences to about $10^{-10}$,
while a backward pass that overwrites instead of accumulating adjoints on
reused nodes produces relative errors of order $0.1$–$2$.
