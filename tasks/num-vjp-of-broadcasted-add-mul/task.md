## Context

When `a` and `b` have different but broadcastable shapes, `out = a + b`
(or `a * b`) implicitly **stretches** each of them up to
`broadcast_shapes((len(a),), (len(b),))`: dimensions of size 1 are
repeated, and missing leading dimensions are inserted. Every one of
those repeats is really a *sum* in the forward pass ($x$ used $k$ times
contributes $k\cdot x$), so backward through a broadcast is a *sum*
gradient reduction — commonly called **sum-to-shape**.

Concretely, given the upstream gradient $\bar y = \partial L/\partial
\text{out}$ (shape = the broadcast output shape), the gradient flowing
back into an input $x$ of shape $\text{shape}(x)$ is obtained from
$\bar y$ by:

1. summing over every axis that broadcasting *added* (extra leading
   axes present in $\bar y$ but not in $x$), then
2. summing (with `keepdims=True`) over every axis where $x$ has size 1
   but $\bar y$ does not,

so that the result has exactly `shape(x)`.

For `out = a + b`: $\bar a = \text{sum\_to\_shape}(\bar y,\ \text{shape}(a))$,
and likewise for $b$.

For `out = a * b`: by the product rule, $\partial \text{out}/\partial a
= b$ and $\partial \text{out}/\partial b = a$ (broadcast the same way as
the forward pass), so
$\bar a = \text{sum\_to\_shape}(\bar y \odot b,\ \text{shape}(a))$ and
$\bar b = \text{sum\_to\_shape}(\bar y \odot a,\ \text{shape}(b))$,
where $\odot$ is the (broadcasting) elementwise product.

## Task

Implement the two VJPs:

```python
def add_vjp(a: list, b: list, grad_out: list):
    ...

def mul_vjp(a: list[float], b: list[float], grad_out: list[float]):
    ...
```

* `a`, `b` — list with broadcastable (possibly different) shapes.
* `grad_out` — the upstream gradient, with shape
`broadcast_shapes((len(a),), (len(b),)).`
* Both return `(grad_a, grad_b)`, where `grad_a.shape == a.shape` and
  `grad_b.shape == b.shape` exactly — do not just reshape or slice
  `grad_out`; the broadcast axes must actually be summed.

## Example

```python
a = [[0.0] * 1 for _ in range(3)]          # shape (3, 1)
b = [[0.0] * 4 for _ in range(1)]          # shape (1, 4)
grad_out = [[1.0] * 4 for _ in range(3)]    # upstream grad, shape (3, 4)

ga, gb = add_vjp(a, b, grad_out)
print(ga.shape, gb.shape)     # (3, 1) (1, 4)
print(ga)                     # each row summed over the 4 broadcast columns -> all 4.0
print(gb)                     # each column summed over the 3 broadcast rows  -> all 3.0
```

## What the gate checks

For five broadcastable shape pairs and both operations, the grader
compares your `grad_a` / `grad_b` against a **central finite-difference**
gradient of `L(a, b) = sum(grad_out * (a OP b))` with respect to every
element of `a` and `b`. By the chain rule this scalar's gradient is
exactly the VJP, so it is a fully independent oracle.

* **max_abs_err** — worst-case entrywise error across every shape pair,
  operation, and array, against the finite-difference oracle. Must be
  below `1e-5`.
