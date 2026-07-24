## Context

NumPy-style broadcasting rules determine whether two array shapes can participate in
an elementwise operation and what the resulting shape is. Given two shapes
$a = (a_1, \ldots, a_m)$ and $b = (b_1, \ldots, b_n)$, the algorithm is:

1. Left-pad the shorter shape with $1$s until both have the same length $d$.
2. For each dimension $i \in \{1, \ldots, d\}$, compare $a_i$ and $b_i$:
   - If $a_i = b_i$, the output dimension is $a_i$.
   - If $a_i = 1$, the output dimension is $b_i$.
   - If $b_i = 1$, the output dimension is $a_i$.
   - Otherwise the shapes are **incompatible** for broadcasting.
3. If no incompatible pair is found, the output shape is
   $\bigl(\max(a_1, b_1),\; \ldots,\; \max(a_d, b_d)\bigr)$.

Scalar shapes are represented as the empty tuple `()`.

## Task

Implement the function `broadcast_shape(shape_a, shape_b)` that classifies whether
two array shapes are broadcast-compatible and returns the result.

- **Input:** `shape_a` and `shape_b`, each a tuple of positive integers (or empty for
  a scalar).
- **Output:** A tuple of integers representing the broadcast result shape if the two
  shapes are compatible, **or** the string `"incompatible"` if they are not.

Do not import NumPy — implement the broadcasting logic from scratch.

## Example

```python
broadcast_shape((3, 4), (1, 4))       # -> (3, 4)
broadcast_shape((5, 1, 3), (1, 4, 1)) # -> (5, 4, 3)
broadcast_shape((2, 3), (5, 4))       # -> "incompatible"
broadcast_shape((), (3,))             # -> (3,)
broadcast_shape((2,), (1, 5))         # -> "incompatible"
```

In the last example, `(2,)` pads to `(1, 2)`. Comparing `(1, 2)` vs `(1, 5)`:
dimension 0 gives $1$, dimension 1 has $2 \neq 5$ and neither is $1$,
so the shapes are incompatible.

## What the gate checks

The gate runs 20 test cases spanning same-rank shapes, different-rank shapes,
scalar inputs, single-element dimensions, and incompatible pairs. For each case it
compares your output to the reference broadcast shape computed from the canonical
algorithm above.

The metric `exact_match` is the fraction of test cases where your answer matches
the reference exactly. The gate passes when `exact_match >= 1.0` — every case must
be correct.
