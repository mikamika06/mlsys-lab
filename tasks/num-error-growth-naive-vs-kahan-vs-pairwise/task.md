## Context

Floating point addition is not exact. When a value is added to a much larger
partial sum, low-order bits can be rounded away. For a sequence
$x_1, \dots, x_N$, ordinary summation computes

$$
s_{k+1} = \operatorname{fl}(s_k + x_{k+1}),
$$

where $\operatorname{fl}$ represents floating point rounding.

Kahan summation stores a compensation value for lost low-order bits:

$$
y = x_i - c,
$$

$$
t = s + y,
$$

$$
c = (t-s)-y.
$$

Pairwise summation changes the order of additions by recursively splitting the
sequence into smaller parts. This reduces the depth of error accumulation from
$O(N)$ additions in a chain to approximately $O(\log N)$ levels.

The error of an approximate sum $\hat{s}$ is measured against a high precision
reference sum $s_{\mathrm{ref}}$:

$$
\mathrm{rel\_err} =
\frac{|\hat{s}-s_{\mathrm{ref}}|}
{|s_{\mathrm{ref}}|+10^{-30}}.
$$

## Task

Implement `summation_error_growth()`:

```python
def summation_error_growth() -> dict:
    ...
```

Return a dictionary with:

- `N`: a NumPy integer array containing the tested sequence lengths.
- `naive`: relative errors from left-to-right summation.
- `kahan`: relative errors from compensated Kahan summation.
- `pairwise`: relative errors from recursive pairwise summation.
- `slopes`: the fitted log-log slopes for the three error arrays in the same
  order.

Use the fixed sequence for each length $N$:

$$
x_1 = 10^{16},
$$

$$
x_i = 1.0 \quad \text{for } i > 1.
$$

For each $N$, evaluate the first $N$ values of this sequence. Use a
high-accuracy summation method internally to obtain the reference value. The
returned measurements should demonstrate how error growth differs between the
three algorithms.

## Example

```python
result = summation_error_growth()

print(result["N"])
# [1000 2000 4000 8000 16000]

print(result["slopes"])
# approximately [1.0, 0.0, 0.0]
```

## What the gate checks

The grader computes its own reference values using a high-accuracy summation
oracle.

The `slope_behavior` gate checks that the measured log-log error slope for
naive summation is close to linear growth and larger than the slopes for Kahan
and pairwise summation.

The `final_error_quality` gate checks the largest sequence length against the
oracle-computed relative errors and verifies that the reported slopes match a
fresh log-log fit.
