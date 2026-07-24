## Context

Floating point arithmetic is not associative because each operation rounds its
result to the representable format. For floating point values $a$, $b$, and $c$,
the two evaluation orders

$$
(a+b)+c
$$

and

$$
a+(b+c)
$$

can produce different bit patterns even though real-number arithmetic would
treat them as equal.

In IEEE 754 single precision, each operation rounds to the nearest representable
`float32` value. This task focuses on constructing an example where rounding
changes the result. The returned values are interpreted as `numpy.float32`
values, so the comparison must consider the actual stored bits.

## Task

Implement `construct_nonassoc_triple()`:

```python
def construct_nonassoc_triple() -> tuple[np.float32, np.float32, np.float32]:
    ...
```

Return three `numpy.float32` values $(a,b,c)$ such that the two float32
computations

$$
x = \operatorname{float32}(\operatorname{float32}(a+b)+c)
$$

and

$$
y = \operatorname{float32}(a+\operatorname{float32}(b+c))
$$

have different IEEE 754 bit patterns.

The function should return the values directly. Do not return the computed
results or any extra metadata.

## Example

```python
a, b, c = construct_nonassoc_triple()

left = np.float32(np.float32(a + b) + c)
right = np.float32(a + np.float32(b + c))

# left and right have different float32 bit patterns
```

## What the gate checks

The gate uses NumPy float32 arithmetic as the reference oracle. It verifies that
the returned triple exists in the float32 format and that the two evaluation
orders produce different stored bit patterns. A triple that does not demonstrate
non-associativity fails the gate.
