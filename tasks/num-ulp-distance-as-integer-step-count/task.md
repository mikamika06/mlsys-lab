## Context

Floating-point numbers are stored as finite bit patterns rather than exact real
numbers. For IEEE-754 float32 values, adjacent representable values differ by
one step in the ordered sequence of bit patterns. This spacing is called a unit
in the last place, or ULP.

A float32 value $x$ has a bit representation that can be viewed as an unsigned
32-bit integer. However, the normal unsigned ordering does not match numeric
ordering because negative values have the sign bit set. A common conversion
creates a monotonic integer key:

$$
k(x) =
\begin{cases}
\sim b, & \text{if the sign bit of } b \text{ is set} \\
b \oplus 2^{31}, & \text{otherwise}
\end{cases}
$$

where $b$ is the unsigned 32-bit representation of $x$, $\sim$ is bitwise
complement, and $\oplus$ is XOR.

The ULP distance between two float32 values is the number of representable
steps between them:

$$
d(a,b) = |k(a)-k(b)|.
$$

This definition makes values near zero and values across exponent boundaries
behave consistently.

## Task

Implement `ulp_distance(a, b)`:

```python
def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...
```

The inputs are NumPy arrays containing values convertible to `float32` and have
the same shape. Return a NumPy array of the same shape containing the integer
ULP distance between corresponding elements.

The returned dtype must be `uint32`. Use vectorized NumPy operations only.

## Example

```python
import numpy as np

a = np.array([-0.0, 1.0, np.float32(1.0)], dtype=np.float32)
b = np.array([0.0, np.nextafter(np.float32(1.0), np.float32(2.0)), np.float32(1.5)], dtype=np.float32)

d = ulp_distance(a, b)
```

The first element has distance $1$ because the two signed zero bit patterns are
adjacent in the ordered float32 representation. The second element has distance
$1$ because `nextafter` moves by one representable float32 step.

## What the gate checks

The gate computes a reference implementation using NumPy bit-level operations and
compares the returned array exactly. The tested values include negative numbers,
zero crossings, subnormal values, and exponent boundaries.

The metric `exact_match` must equal $1.0$. Any implementation that subtracts
floating-point values or ignores the sign-bit ordering will fail.
