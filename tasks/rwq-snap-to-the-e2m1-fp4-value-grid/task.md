## Context

The E2M1 FP4 format packs a floating-point value into four bits: one sign bit,
one exponent bit, and one mantissa bit.  The representable magnitudes form an
eight-level grid

$$\mathcal{G} = \{0,\; 0.5,\; 1,\; 1.5,\; 2,\; 3,\; 4,\; 6\}$$

and the full signed grid (zero has no sign) is

$$\mathcal{G}_{\text{signed}} = \{-6,\; -4,\; -3,\; -2,\; -1.5,\; -1,\; -0.5,\; 0,\; 0.5,\; 1,\; 1.5,\; 2,\; 3,\; 4,\; 6\}.$$

Post-training quantization maps each real-valued weight or activation $x_i$ to
the nearest representable level via a nearest-grid-point assignment.  Because
$\mathcal{G}_{\text{signed}}$ is symmetric about zero the operation decomposes
into two steps:

1. Compute $|x_i|$ and find the nearest magnitude
   $g^* = \arg\min_{g \in \mathcal{G}} \bigl|\,|x_i| - g\bigr|.$
2. Return $g^* \cdot \operatorname{sign}(x_i).$

Any value $|x_i| > 6$ is clamped to the outermost level $6$.

## Task

Implement `snap_to_e2m1(x)`:

```python

def snap_to_e2m1(x):
    """Snap each element of *x* to the nearest signed E2M1 FP4 value."""
    ...
```

Input: a 1-D `float64` list whose elements lie in approximately
$[-6.5,\, 6.5]$.

Output: a list of the same shape containing the nearest element of
$\mathcal{G}_{\text{signed}}$ for every input value.

## Example

```python
x = [0.3, 0.7, 1.2, 2.7, -5.5]
snap_to_e2m1(x)
# array([ 0.5,  0.5,  1. ,  3. , -6. ])
```

Explanation of each snap:

| $x_i$ | $|x_i|$ | nearest magnitude | result |
|--------|---------|-------------------|--------|
| $0.3$ | $0.3$ | $0.5$ | $0.5$ |
| $0.7$ | $0.7$ | $0.5$ | $0.5$ |
| $1.2$ | $1.2$ | $1.0$ | $1.0$ |
| $2.7$ | $2.7$ | $3.0$ | $3.0$ |
| $-5.5$ | $5.5$ | $6.0$ | $-6.0$ |

## What the gate checks

The gate metric is `max_abs_err`.  The grader constructs an independent Python
oracle that builds the same magnitude grid $\mathcal{G}$, computes
$\arg\min_{g \in \mathcal{G}} \bigl||x_i| - g\bigr|$ via broadcasting, and
reapplies the sign.  It then evaluates several inline test vectors spanning
midpoints, boundaries, zeros, negatives, and near-clamp values.  The maximum
element-wise absolute difference between the learner's output and the oracle
must satisfy

$$\texttt{max\_abs\_err} < 10^{-6}.$$

An incorrect grid, a sign mistake, or a shape mismatch will exceed this
threshold.
