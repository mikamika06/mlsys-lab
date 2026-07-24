## Context

A compiler for tensor programs often represents dimensions symbolically instead of
requiring concrete sizes. A symbolic dimension such as $s_0$ can flow through
operators while the compiler records constraints needed for valid execution.

For a tensor with shape $(d_0, d_1, \dots, d_n)$, the number of elements is

$$
\mathrm{numel} = \prod_i d_i .
$$

A reshape operation preserves this value, so changing

$$
(a,b) \rightarrow (c,d)
$$

requires the guard

$$
a b = c d .
$$

A reshape with a symbolic dimension may introduce divisibility constraints. For
example, reshaping $(s_0, 4)$ into $(8, -1)$ requires

$$
s_0 \bmod 2 = 0
$$

because the known elements after dividing by $8$ must be integral.

This task models a small symbolic shape engine. It propagates expressions through
reshape, view, cat, and matmul operations and emits guards that must hold for all
runtime values.

## Task

Implement `propagate_shapes(graph)`:

```python
def propagate_shapes(graph):
    ...
```

The input is a list of operation dictionaries. Each operation has an `op` field
and names tensors through `inputs` and `output`.

Supported operations:

- `input`: creates a tensor. The dictionary contains `shape`, a list of symbolic
  dimension strings.
- `reshape`: contains `shape`, where `-1` is inferred.
- `view`: behaves like `reshape`.
- `cat`: contains `axis` and concatenates input shapes on that axis.
- `matmul`: supports two-dimensional matrix multiplication.

Return a dictionary:

```python
{
    "shapes": {"tensor_name": "dim0,dim1,..."},
    "guards": ["guard expression", ...]
}
```

Shapes must use comma-separated expressions. Guards are compared as a set after
sorting.

The implementation should derive expressions and constraints from the graph. Do
not evaluate symbolic dimensions.

## Example

```python
graph = [
    {"op": "input", "output": "x", "shape": ["s0", "4"]},
    {"op": "reshape", "inputs": ["x"], "output": "y", "shape": ["8", "-1"]}
]

propagate_shapes(graph)

# {
#   "shapes": {"x": "s0,4", "y": "8,s0*4/8"},
#   "guards": ["s0*4 % 8 == 0"]
# }
```

## What the gate checks

The gate builds reference outputs with an independent symbolic propagation
algorithm. It checks that the returned shape expressions and emitted guards match
the oracle result exactly after guard normalization.

The tested graphs include symbolic dimensions, reshape inference, concatenation,
and matrix multiplication constraints.
