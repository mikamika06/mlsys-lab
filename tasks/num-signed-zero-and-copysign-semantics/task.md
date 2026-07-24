## Context

IEEE 754 floating-point numbers include a signed zero value. The values $+0.0$ and
$-0.0$ compare equal, but they have different sign bits and can produce different
results in operations.

For example, division preserves the sign information:

$$\frac{1}{-0.0} = -\infty,\qquad \frac{1}{+0.0} = +\infty.$$

The sign of zero results can also depend on the operation. Adding opposite signed
zeros follows floating-point rules rather than ordinary real arithmetic, and
functions such as `copysign` explicitly transfer a sign bit.

A reliable program should inspect sign bits directly instead of using equality,
because

$$+0.0 = -0.0$$

is true as a numerical comparison even though the bit-level signs differ.

## Task

Implement `signed_zero_profile()`.

The function takes no arguments and returns a list of integers. Each integer must
be the sign bit (`0` for non-negative and `1` for negative) of the following
NumPy-computed values, in order:

1. `1.0 / -0.0`
2. `-0.0 + 0.0`
3. `0.0 + -0.0`
4. `np.copysign(0.0, -1.0)`
5. `np.copysign(-0.0, 1.0)`
6. `np.copysign(5.0, -0.0)`

Use NumPy floating-point semantics. The returned value must be a Python list
containing exactly six integers.

## Example

```python
result = signed_zero_profile()
# result is a list like [1, 0, 0, 1, 0, 1]
```

The exact sign of expressions involving signed zero must come from the NumPy
oracle rather than from decimal equality checks.

## What the gate checks

The gate computes the expected sign-bit vector using NumPy's `signbit` and compares
it with the returned list.

The `exact_match` score is `1.0` only when every sign bit matches the NumPy
reference. A solution that treats `0.0` and `-0.0` as interchangeable will fail.
